import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

/**
 * One ephemeral EC2 **Spot** instance for benchmarking, in the default VPC
 * (public subnet, no NAT cost). Build/run via SSH, then `cdk destroy`.
 *
 * Override anything with `-c key=value`:
 *   -c keyPairName=NAME   (required) existing EC2 key pair to SSH with
 *   -c instanceType=...   default c8i.2xlarge (Granite Rapids)
 *   -c sshCidr=1.2.3.4/32 default 0.0.0.0/0 (lock to your IP if you can)
 *   -c maxSpotPrice=0.40  optional $/hr cap (default: on-demand price)
 *   -c volumeGb=30        root EBS size
 */
interface SpotStackProps extends cdk.StackProps {
  /** Per-stack instance type (used by the sweep, where context is shared across stacks). */
  instanceTypeOverride?: string;
  /** Public-subnet index to launch in (0/1/2 → AZ a/b/c). Lets us dodge per-AZ
   *  spot-capacity holes (e.g. c8g had none in -1a). Default 0. */
  azIndex?: number;
}

export class SpotStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: SpotStackProps) {
    super(scope, id, props);

    const c = this.node;
    const instanceType: string =
      props?.instanceTypeOverride ?? c.tryGetContext('instanceType') ?? 'c8i.2xlarge';
    const keyPairName: string | undefined = c.tryGetContext('keyPairName');
    const sshCidr: string = c.tryGetContext('sshCidr') ?? '0.0.0.0/0';
    const maxSpotPrice: string | undefined = c.tryGetContext('maxSpotPrice');
    const volumeGb = Number(c.tryGetContext('volumeGb') ?? 30);

    // Key pair: use an existing one if named, otherwise CREATE a fresh one.
    // A created key pair's private key is auto-stored in SSM Parameter Store
    // (SecureString) — fetch it after deploy; we never handle the private key.
    let keyPair: ec2.IKeyPair;
    let createdKeyId: string | undefined;
    if (keyPairName) {
      keyPair = ec2.KeyPair.fromKeyPairName(this, 'Key', keyPairName);
    } else {
      const kp = new ec2.KeyPair(this, 'Key', {
        keyPairName: 'vps-bench',
        type: ec2.KeyPairType.ED25519,
        format: ec2.KeyPairFormat.PEM,
      });
      keyPair = kp;
      createdKeyId = kp.keyPairId;
    }

    // Default VPC: public subnets, no NAT gateway → no idle cost.
    const vpc = ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true });

    const sg = new ec2.SecurityGroup(this, 'Sg', {
      vpc,
      description: 'vps spot benchmark ssh',
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.ipv4(sshCidr), ec2.Port.tcp(22), 'ssh');

    const userData = ec2.UserData.forLinux();
    // System-level deps as root; Rust is installed per-user by run-on-aws.sh.
    userData.addCommands('dnf install -y git gcc perf || true');

    // Pick the AMI arch from the instance family: the processor letter is the
    // char right after the generation digit(s) — 'g' = Graviton (arm64),
    // 'i' = Intel, 'a' = AMD (both x86_64). e.g. c8g→arm64, c8a/m8i→x86_64.
    const procLetter = (instanceType.split('.')[0].match(/^[a-z]+\d+([a-z])/)?.[1]) ?? 'i';
    const cpuType =
      procLetter === 'g' ? ec2.AmazonLinuxCpuType.ARM_64 : ec2.AmazonLinuxCpuType.X86_64;

    // GPU families (g5/g6/p*) need NVIDIA drivers, which plain AL2023 lacks — use the
    // AWS Deep Learning base AMI (AL2023 + NVIDIA driver + CUDA). x86_64 only.
    const isGpu = /^(g|p)\d/.test(instanceType);
    const machineImage = isGpu
      ? ec2.MachineImage.fromSsmParameter(
          '/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id',
          { os: ec2.OperatingSystemType.LINUX },
        )
      : ec2.MachineImage.latestAmazonLinux2023({ cpuType });

    const lt = new ec2.LaunchTemplate(this, 'Lt', {
      instanceType: new ec2.InstanceType(instanceType),
      machineImage,
      keyPair,
      userData,
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(volumeGb, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        },
      ],
      // One-time spot request: if interrupted it's gone (fine for ephemeral
      // benchmarking — no auto-replace surprises).
      spotOptions: {
        requestType: ec2.SpotRequestType.ONE_TIME,
        interruptionBehavior: ec2.SpotInstanceInterruption.TERMINATE,
        ...(maxSpotPrice ? { maxPrice: Number(maxSpotPrice) } : {}),
      },
    });

    const azIndex = props?.azIndex ?? Number(c.tryGetContext('azIndex') ?? 0);
    const subnet = vpc.publicSubnets[azIndex % vpc.publicSubnets.length];

    const instance = new ec2.CfnInstance(this, 'Spot', {
      launchTemplate: {
        launchTemplateId: lt.launchTemplateId!,
        version: lt.latestVersionNumber,
      },
      subnetId: subnet.subnetId,
      securityGroupIds: [sg.securityGroupId],
      tags: [{ key: 'Name', value: 'vps-spot-bench' }],
    });

    new cdk.CfnOutput(this, 'InstanceType', { value: instanceType });
    new cdk.CfnOutput(this, 'PublicIp', { value: instance.attrPublicIp });
    new cdk.CfnOutput(this, 'Ssh', {
      value: `ssh ec2-user@${instance.attrPublicIp}`,
    });
    if (createdKeyId) {
      new cdk.CfnOutput(this, 'FetchPrivateKey', {
        value:
          `aws ssm get-parameter --name /ec2/keypair/${createdKeyId} ` +
          `--with-decryption --query Parameter.Value --output text ` +
          `--region ${this.region} > ~/.ssh/vps-bench.pem && chmod 600 ~/.ssh/vps-bench.pem`,
      });
    }
  }
}
