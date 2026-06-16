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
export class SpotStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const c = this.node;
    const instanceType: string = c.tryGetContext('instanceType') ?? 'c8i.2xlarge';
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

    const lt = new ec2.LaunchTemplate(this, 'Lt', {
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.latestAmazonLinux2023({
        cpuType: ec2.AmazonLinuxCpuType.X86_64,
      }),
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

    const instance = new ec2.CfnInstance(this, 'Spot', {
      launchTemplate: {
        launchTemplateId: lt.launchTemplateId!,
        version: lt.latestVersionNumber,
      },
      subnetId: vpc.publicSubnets[0].subnetId,
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
