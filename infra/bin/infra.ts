#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { SpotStack } from '../lib/spot-stack';

const app = new cdk.App();

new SpotStack(app, 'VpsSpotStack', {
  // Region defaults to Singapore (where you are → low SSH latency). Override
  // with CDK_DEFAULT_REGION or `-c region=...`. Account comes from your creds.
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region:
      app.node.tryGetContext('region') ??
      process.env.CDK_DEFAULT_REGION ??
      'ap-southeast-1',
  },
});
