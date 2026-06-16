# infra — CDK for ephemeral spot benchmark boxes

One EC2 **Spot** instance in the default VPC, for running the benchmarks on a
specific CPU, then tearing down. Default: `c8i.2xlarge` (Granite Rapids) in
`ap-southeast-1` (Singapore — low SSH latency from here).

## One-time setup

```bash
cd infra
npm install

# AWS credentials. This account uses SSO (IAM Identity Center):
aws sso login --profile <your-profile>
export AWS_PROFILE=<your-profile> AWS_REGION=ap-southeast-1

# Bootstrap CDK once per account+region
npx cdk bootstrap
```

## Launch / tear down

By default the stack **creates a fresh key pair** (`vps-bench`) and stores its
private key in SSM — you fetch it once; the private key never passes through
this tool.

```bash
# launch (prints PublicIp + a FetchPrivateKey command as stack outputs)
npx cdk deploy --require-approval never

# fetch the new private key (command is in the deploy output):
aws ssm get-parameter --name /ec2/keypair/<id> --with-decryption \
  --query Parameter.Value --output text --region ap-southeast-1 \
  > ~/.ssh/vps-bench.pem && chmod 600 ~/.ssh/vps-bench.pem

# ssh:  ssh -i ~/.ssh/vps-bench.pem ec2-user@<PublicIp>
# ... run benchmarks (rsync repo, bash history/run-on-aws.sh) ...

# DESTROY when done — this is how you keep it cheap
npx cdk destroy
```

To use an **existing** key pair instead of creating one: `-c keyPairName=NAME`.

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
