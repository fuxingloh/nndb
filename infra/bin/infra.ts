#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { SpotStack } from '../lib/spot-stack';

const app = new cdk.App();

const env = {
  // Region defaults to Singapore (where you are → low SSH latency). Override
  // with CDK_DEFAULT_REGION or `-c region=...`. Account comes from your creds.
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region:
    app.node.tryGetContext('region') ??
    process.env.CDK_DEFAULT_REGION ??
    'ap-southeast-1',
};

// Two ways to drive this:
//   single : cdk deploy -c instanceType=c8g.2xlarge -c keyPairName=vps-bench
//   sweep  : cdk deploy --all -c sweep=c8i.2xlarge,c8a.2xlarge,... -c keyPairName=vps-bench
// Each instance type gets its own stack "VpsSpot-<type>" (so the 6 are independent
// and can be destroyed individually). `sweep` wins if both are set.
const sweep: string | undefined = app.node.tryGetContext('sweep');
const single: string | undefined = app.node.tryGetContext('instanceType');
const types = sweep
  ? sweep.split(',').map((s: string) => s.trim()).filter(Boolean)
  : [single]; // [undefined] → one stack on the SpotStack default (c8i.2xlarge)

for (const it of types) {
  const stackName = it ? `VpsSpot-${it.replace(/[^a-z0-9]/gi, '')}` : 'VpsSpotStack';
  new SpotStack(app, stackName, { env, instanceTypeOverride: it });
}
