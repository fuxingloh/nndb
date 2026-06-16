# infra — CDK for ephemeral spot benchmark boxes

One EC2 **Spot** instance in the default VPC, for running the benchmarks on a
specific CPU, then tearing down. Default: `c8i.2xlarge` (Granite Rapids) in
`ap-southeast-1` (Singapore — low SSH latency from here).

## One-time setup

```bash
cd infra
npm install

# 1. AWS credentials (CDK needs them; the AWS CLI had none configured)
aws configure                      # or export AWS_ACCESS_KEY_ID / SECRET

# 2. Register an SSH key pair EC2 can use. Import your EXISTING public key so
#    you keep using your normal private key (we never touch the private key):
aws ec2 import-key-pair --key-name vps \
  --public-key-material fileb://~/.ssh/id_ed25519.pub --region ap-southeast-1

# 3. Bootstrap CDK once per account+region
npx cdk bootstrap
```

## Launch / tear down

```bash
# launch (prints the public IP + ssh line as stack outputs)
npx cdk deploy -c keyPairName=vps

# ... run benchmarks over SSH (rsync repo, bash history/run-on-aws.sh) ...

# DESTROY when done — this is how you keep it cheap
npx cdk destroy
```

Useful overrides:

```bash
npx cdk deploy -c keyPairName=vps -c instanceType=c7i.2xlarge   # Sapphire Rapids
npx cdk deploy -c keyPairName=vps -c instanceType=c8a.2xlarge   # Zen 5
npx cdk deploy -c keyPairName=vps -c sshCidr=$(curl -s ifconfig.me)/32  # lock SSH to your IP
npx cdk deploy -c keyPairName=vps -c maxSpotPrice=0.40 -c region=us-east-2
```

## Notes

- **Cost:** spot `c8i.2xlarge` ≈ $0.22–0.30/hr. A `cdk destroy` removes the
  instance, SG, and launch template — nothing keeps billing. The root EBS is
  deleted with the instance.
- **`c8i` is new** — if the spot request can't be fulfilled in your region/AZ,
  fall back to `-c instanceType=c7i.2xlarge` (Sapphire Rapids, broad capacity).
- **Spot = one-time, terminate-on-interruption** — if AWS reclaims it the box
  is gone (acceptable for ephemeral runs; just redeploy).
- **No PMU** on virtualized instances regardless of type — use the software
  bound detector (`history/measure-bound.sh`), not `perf` top-down.
- Region only affects price + your SSH latency, never the benchmark result.
