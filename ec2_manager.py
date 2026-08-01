"""
EC2 instance management for this project's compute fleet - launch (Spot or
on-demand, with automatic AZ retry since Spot capacity for large instance
types has repeatedly been AZ-specific this session), resize, stop, and
terminate. Consolidates what was otherwise a dozen ad-hoc boto3 one-liners
into one place.

Resize note: an on-demand (EBS-backed) instance can genuinely be resized in
place - stop, change InstanceType, start, same disk. A Spot "one-time"
request cannot: AWS has no API to change a running Spot instance's type, so
"resize" there means terminate and launch a fresh Spot request at the new
size. That loses anything not already durable (this project's convention is
S3 as the durable store and local disk as scratch, so that's normally fine,
but a resize mid-AC loses that AC's in-progress OCR pages).

Usage
-----
    python ec2_manager.py list
    python ec2_manager.py launch --name foo --type c7i.8xlarge --market spot
    python ec2_manager.py launch --name foo --type c7i.4xlarge --market on-demand --no-bootstrap
    python ec2_manager.py resize --name foo --type c7i.16xlarge
    python ec2_manager.py stop --name foo
    python ec2_manager.py start --name foo
    python ec2_manager.py terminate --name foo
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import boto3

REGION = "ap-south-1"
SUBNETS = {  # ap-south-1a/1b/1c - tried in this order, since Spot capacity
             # for large instance types has been AZ-specific this session
    "ap-south-1a": "subnet-0dd8a9ddb7ddf91e9",
    "ap-south-1b": "subnet-060bc92671e111340",
    "ap-south-1c": "subnet-02e195e9e4b3303cc",
}
SECURITY_GROUP = "sg-04c75bb8ecec85585"  # default SG: allows all traffic within
                                         # itself + inbound SSH from anywhere
DEFAULT_KEY = "voter-pipeline-spot"
AMI_X86 = "ami-07e5ce642bbc48c0d"    # ubuntu-noble-24.04-amd64
AMI_ARM = "ami-03f419e5dee7ea8f3"    # ubuntu-noble-24.04-arm64
ARM_TYPE_PREFIXES = ("c6g.", "c7g.", "m6g.", "m7g.", "r6g.", "r7g.", "t4g.")

BOOTSTRAP_USERDATA = """#!/usr/bin/env bash
exec > /var/log/bootstrap.log 2>&1
apt-get update -qq
apt-get install -y -qq tesseract-ocr tesseract-ocr-hin python3-venv python3-pip git \\
    libgl1 libglib2.0-0
sudo -u ubuntu bash -c '
cd /home/ubuntu
git clone -q https://github.com/tejaspawar18/booth_list.git
cd booth_list
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt pytesseract boto3
mkdir -p tessdata_best
curl -sL -o tessdata_best/eng.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata
'
echo BOOTSTRAP_DONE
"""


def ec2():
    return boto3.client("ec2", region_name=REGION)


def find_by_name(name: str) -> dict | None:
    r = ec2().describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [name]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    instances = [i for res in r["Reservations"] for i in res["Instances"]]
    if not instances:
        return None
    if len(instances) > 1:
        print(f"warning: {len(instances)} instances named {name!r}, using the most recent",
              file=sys.stderr)
    return sorted(instances, key=lambda i: i["LaunchTime"])[-1]


def cmd_list(_args: argparse.Namespace) -> None:
    r = ec2().describe_instances(Filters=[
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    eip_by_instance = {a["InstanceId"]: a["PublicIp"]
                       for a in ec2().describe_addresses()["Addresses"] if a.get("InstanceId")}
    rows = []
    for res in r["Reservations"]:
        for i in res["Instances"]:
            name = next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "(unnamed)")
            pub = i.get("PublicIpAddress", "")
            is_eip = i["InstanceId"] in eip_by_instance
            pub_label = f"{pub} (elastic)" if pub and is_eip else pub or "-"
            rows.append((name, i["InstanceId"], i["InstanceType"],
                        i.get("InstanceLifecycle", "on-demand"), i["State"]["Name"],
                        i.get("PrivateIpAddress", ""), pub_label))
    if not rows:
        print("no instances running")
        return
    w = max(len(r[0]) for r in rows)
    for name, iid, itype, lifecycle, state, priv_ip, pub_label in sorted(rows):
        print(f"{name:<{w}}  {iid}  {itype:<15}  {lifecycle:<10}  {state:<10}  "
             f"private={priv_ip:<15}  public={pub_label}")


def ensure_eip(instance_id: str, name: str) -> str:
    """Idempotent: if the instance already has an Elastic IP, return it as-is;
    otherwise allocate and associate a new one. An EIP survives stop/start/
    resize (only terminating the instance or explicitly releasing it drops
    it) - this is what makes a box's address permanent instead of a fresh
    ephemeral public IP on every restart."""
    existing = ec2().describe_addresses(Filters=[{"Name": "instance-id", "Values": [instance_id]}])
    addrs = existing["Addresses"]
    if addrs:
        return addrs[0]["PublicIp"]
    alloc = ec2().allocate_address(
        Domain="vpc", TagSpecifications=[{"ResourceType": "elastic-ip",
                                         "Tags": [{"Key": "Name", "Value": f"{name}-eip"}]}])
    ec2().associate_address(InstanceId=instance_id, AllocationId=alloc["AllocationId"])
    return alloc["PublicIp"]


def cmd_eip(args: argparse.Namespace) -> None:
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    ip = ensure_eip(inst["InstanceId"], args.name)
    print(f"{args.name} ({inst['InstanceId']}): static public IP {ip}")


def cmd_launch(args: argparse.Namespace) -> None:
    if find_by_name(args.name):
        raise SystemExit(f"an instance named {args.name!r} already exists - pick another "
                         f"name or terminate it first")
    is_arm = args.type.startswith(ARM_TYPE_PREFIXES)
    ami = args.ami or (AMI_ARM if is_arm else AMI_X86)
    userdata = "" if args.no_bootstrap else BOOTSTRAP_USERDATA

    kwargs = dict(
        ImageId=ami, InstanceType=args.type, MinCount=1, MaxCount=1,
        KeyName=args.key, SecurityGroupIds=[SECURITY_GROUP],
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": args.disk_gb, "VolumeType": "gp3"}}],
        TagSpecifications=[{"ResourceType": "instance",
                           "Tags": [{"Key": "Name", "Value": args.name}]}],
        UserData=userdata,
    )
    if args.market == "spot":
        kwargs["InstanceMarketOptions"] = {"MarketType": "spot",
                                          "SpotOptions": {"SpotInstanceType": "one-time"}}

    last_error = None
    for az, subnet in SUBNETS.items():
        try:
            r = ec2().run_instances(SubnetId=subnet, **kwargs)
            iid = r["Instances"][0]["InstanceId"]
            print(f"launched {args.name} ({iid}) in {az}, security group {SECURITY_GROUP}")
            if args.eip:
                ip = ensure_eip(iid, args.name)
                print(f"static public IP: {ip}")
            if args.wait:
                ec2().get_waiter("instance_running").wait(InstanceIds=[iid])
                priv_ip = ec2().describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
                print(f"running, private ip {priv_ip} - connect from this box with: "
                     f"python ec2_manager.py ssh --name {args.name}")
                if userdata:
                    print("bootstrap running in the background - poll /var/log/bootstrap.log "
                         "for BOOTSTRAP_DONE before using the box")
            return
        except Exception as exc:  # noqa: BLE001 - try the next AZ regardless of why this one failed
            last_error = exc
            print(f"{az} failed: {str(exc)[:150]}", file=sys.stderr)
    raise SystemExit(f"launch failed in all AZs; last error: {last_error}")


def cmd_resize(args: argparse.Namespace) -> None:
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    iid = inst["InstanceId"]
    if inst.get("InstanceLifecycle") == "spot":
        raise SystemExit(
            f"{args.name} ({iid}) is a Spot one-time instance - AWS has no in-place resize "
            f"for these. Terminate it and launch a new one at {args.type} instead:\n"
            f"  python ec2_manager.py terminate --name {args.name}\n"
            f"  python ec2_manager.py launch --name {args.name} --type {args.type} --market spot")

    current = inst["InstanceType"]
    if current == args.type:
        print(f"{args.name} is already {args.type}, nothing to do")
        return

    print(f"stopping {args.name} ({iid}, {current} -> {args.type})...")
    ec2().stop_instances(InstanceIds=[iid])
    ec2().get_waiter("instance_stopped").wait(InstanceIds=[iid])
    ec2().modify_instance_attribute(InstanceId=iid, InstanceType={"Value": args.type})
    print("starting...")
    ec2().start_instances(InstanceIds=[iid])
    if args.wait:
        ec2().get_waiter("instance_running").wait(InstanceIds=[iid])
        ip = ec2().describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        print(f"running as {args.type}, private ip {ip}")
    else:
        print(f"resize to {args.type} underway")


def cmd_stop(args: argparse.Namespace) -> None:
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    if inst.get("InstanceLifecycle") == "spot":
        raise SystemExit(f"{args.name} is a Spot one-time instance - these can't be stopped "
                         f"and resumed, only terminated (python ec2_manager.py terminate "
                         f"--name {args.name})")
    ec2().stop_instances(InstanceIds=[inst["InstanceId"]])
    print(f"stopping {args.name} ({inst['InstanceId']})")
    if args.wait:
        ec2().get_waiter("instance_stopped").wait(InstanceIds=[inst["InstanceId"]])
        print("stopped")


def cmd_start(args: argparse.Namespace) -> None:
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    ec2().start_instances(InstanceIds=[inst["InstanceId"]])
    print(f"starting {args.name} ({inst['InstanceId']})")
    if args.wait:
        ec2().get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
        ip = ec2().describe_instances(InstanceIds=[inst["InstanceId"]])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        print(f"running, private ip {ip}")


def cmd_terminate(args: argparse.Namespace) -> None:
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    if not args.yes:
        confirm = input(f"terminate {args.name} ({inst['InstanceId']}, "
                        f"{inst['InstanceType']})? [y/N] ")
        if confirm.strip().lower() != "y":
            print("cancelled")
            return
    ec2().terminate_instances(InstanceIds=[inst["InstanceId"]])
    print(f"terminating {args.name} ({inst['InstanceId']})")


def cmd_ssh(args: argparse.Namespace) -> None:
    """Resolve the instance's current private IP live and exec straight into
    ssh - no config file to go stale. For connecting *from this box* (same
    VPC, so private IP; for a laptop outside the VPC use `eip` instead and
    add a Host entry there, since a private IP is unreachable from outside)."""
    inst = find_by_name(args.name)
    if not inst:
        raise SystemExit(f"no instance named {args.name!r}")
    ip = inst.get("PrivateIpAddress")
    if not ip:
        raise SystemExit(f"{args.name} ({inst['InstanceId']}) has no private IP right now "
                         f"(state: {inst['State']['Name']}) - start it first")
    sgs = [g["GroupId"] for g in inst.get("SecurityGroups", [])]
    if SECURITY_GROUP not in sgs:
        print(f"warning: {args.name} is not in this project's shared security group "
             f"{SECURITY_GROUP} (has {sgs}) - if you're connecting from another instance "
             f"rather than this one, the connection may be refused", file=sys.stderr)
    key = inst.get("KeyName", DEFAULT_KEY)
    key_path = os.path.expanduser(f"~/.ssh/{key}.pem")
    print(f"connecting to {args.name} ({inst['InstanceId']}) at {ip} via security "
         f"group {sgs}", file=sys.stderr)
    os.execvp("ssh", ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=accept-new",
                      f"ubuntu@{ip}", *args.command])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show all instances and their state")

    p = sub.add_parser("launch", help="create a new instance")
    p.add_argument("--name", required=True)
    p.add_argument("--type", required=True, help="e.g. c7i.8xlarge, c7i.4xlarge, c6g.8xlarge")
    p.add_argument("--market", choices=["spot", "on-demand"], default="spot")
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--ami", help="default: Ubuntu 24.04, arch auto-detected from --type")
    p.add_argument("--disk-gb", type=int, default=60)
    p.add_argument("--no-bootstrap", action="store_true",
                   help="skip installing tesseract/venv/repo via user-data")
    p.add_argument("--eip", action="store_true",
                   help="allocate + associate a static Elastic IP (for connecting from "
                        "outside the VPC, e.g. a laptop - survives stop/start/resize)")
    p.add_argument("--no-wait", dest="wait", action="store_false",
                   help="return as soon as the launch API call succeeds, don't wait for running")

    p = sub.add_parser("resize", help="change an on-demand instance's type (stop/modify/start)")
    p.add_argument("--name", required=True)
    p.add_argument("--type", required=True)
    p.add_argument("--no-wait", dest="wait", action="store_false")

    p = sub.add_parser("stop", help="stop an on-demand instance (keeps disk, no compute billing)")
    p.add_argument("--name", required=True)
    p.add_argument("--no-wait", dest="wait", action="store_false")

    p = sub.add_parser("start", help="start a stopped instance")
    p.add_argument("--name", required=True)
    p.add_argument("--no-wait", dest="wait", action="store_false")

    p = sub.add_parser("terminate", help="permanently destroy an instance")
    p.add_argument("--name", required=True)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    p = sub.add_parser("ssh", help="connect from this box via the instance's live private IP")
    p.add_argument("--name", required=True)
    p.add_argument("command", nargs="*", help="remote command to run instead of an interactive shell")

    p = sub.add_parser("eip", help="ensure a static Elastic IP is associated (idempotent)")
    p.add_argument("--name", required=True)

    args = ap.parse_args()
    {"list": cmd_list, "launch": cmd_launch, "resize": cmd_resize, "stop": cmd_stop,
     "start": cmd_start, "terminate": cmd_terminate, "ssh": cmd_ssh,
     "eip": cmd_eip}[args.cmd](args)


if __name__ == "__main__":
    main()
