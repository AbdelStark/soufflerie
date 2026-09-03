# Validation status: RED

- Overall status: **RED**
- Report ID: `4995ef8f8456030f467d`
- Dataset ID: `4aefbbe88a18d233249b`
- Selected model ID: `7b6fd39c0ec78f452163`
- Ensemble model IDs: `7b6fd39c0ec78f452163`, `8847109ea4eaf87aa5ae`, `bec20fbbb9dfd64f0639`
- Baseline IDs: `66990bf5626da7eb4c10`, `3f8924d130bb0b22a7ad`
- Generator: `validation-report-v1`

> **Release blocked:** one or more required gates are red. The plots below
> are diagnostic evidence and do not override this status.

## Required gates

| Gate | Status | Value | Operator | Threshold | Units | Evidence |
|---|---|---:|:---:|---:|---|---|
| field_error | GREEN | 0.00632138 | lt | 0.08 | ratio | test:200 |
| cd_head_error | GREEN | 1.86444 | lt | 5 | percent | test:200 |
| head_field_consistency | RED | 0.615 | ge | 0.95 | fraction | cases:200; condition:head_field_gap_pct<=10.0 |
| divergence | RED | 5.72436 | lt | 3 | ratio | prediction_count:200; solver_count:200 |
| obstacle_compliance | RED | 0.98189 | lt | 0.01 | ratio | test:200 |
| mean_baseline_field | GREEN | 0.00632138 | lt | 0.0402291 | ratio | same-test-membership:200 |
| nearest_baseline_field | GREEN | 0.00632138 | lt | 0.0119388 | ratio | same-test-membership:200 |
| mean_baseline_cd | GREEN | 1.86444 | lt | 17.3283 | percent | same-test-membership:200 |
| nearest_baseline_cd | GREEN | 1.86444 | lt | 4.88008 | percent | same-test-membership:200 |
| ood_variance_increase | GREEN | 3.49486 | ge | 1.5 | ratio | model_ids:7b6fd39c0ec78f452163,8847109ea4eaf87aa5ae,bec20fbbb9dfd64f0639; probe_count:40 |
| sensitivity_sign | RED | 4 | ge | 8 | count_of_10 | model_id:7b6fd39c0ec78f452163; probe_count:10 |
| evidence_integrity | GREEN | true | eq | true | boolean | all-parent-digests-and-fixed-memberships-verified |

## Metric distributions

| Metric | Status | Count | Median | P90 | P95 | Maximum | Bootstrap median 95% |
|---|---|---:|---:|---:|---:|---:|---|
| mean.cd_head_pct | valid | 200 | 17.3283 | 40.5044 | 52.5233 | 84.1887 | 14.3343 - 20.4665 |
| mean.velocity_rel_l2 | valid | 200 | 0.0402291 | 0.0712665 | 0.0852351 | 0.131718 | 0.0367699 - 0.0425312 |
| nearest.cd_head_pct | valid | 200 | 4.88008 | 14.2969 | 18.2994 | 33.1078 | 3.94117 - 6.00653 |
| nearest.velocity_rel_l2 | valid | 200 | 0.0119388 | 0.0288544 | 0.0346341 | 0.0948934 | 0.0109102 - 0.0138615 |
| selected.cd_field_pct | valid | 200 | 6.98004 | 26.557 | 31.938 | 43.5375 | 5.7744 - 8.99774 |
| selected.cd_head_pct | valid | 200 | 1.86444 | 4.75579 | 5.36011 | 17.9536 | 1.64365 - 2.02792 |
| selected.head_field_gap_pct | valid | 200 | 7.6683 | 27.2511 | 31.9643 | 42.0646 | 6.61303 - 9.32007 |
| selected.obstacle_ratio | valid | 200 | 0.960732 | 0.976518 | 0.98189 | 0.991159 | 0.959338 - 0.963245 |
| selected.prediction_div_mean_abs | valid | 200 | 2.15479e-05 | 3.29633e-05 | 3.77456e-05 | 4.74709e-05 | 2.07655e-05 - 2.23885e-05 |
| selected.solver_div_mean_abs | valid | 200 | 3.76424e-06 | 3.92099e-06 | 3.98758e-06 | 4.11031e-06 | 3.74671e-06 - 3.78372e-06 |
| selected.velocity_rel_l2 | valid | 200 | 0.00632138 | 0.0121445 | 0.0146629 | 0.0538239 | 0.00587851 - 0.00679713 |

Worst `mean.cd_head_pct` cases: `6a664f2737170d5566ad`, `1ec2cbc77735b412a6da`, `49d25c943ef8b1339c2c`, `ca545a9f4226e7241a4c`, `97da78f942cbbb511f88`, `4891c9d350a47daed666`, `851cf9601aed2020f5d1`, `0224e55d2f834fcee90d`, `a8e5ea6ab49b95adc036`, `b12971de8b3edcbf49b7`, `ece00f37916f23ff2fd2`, `c13deb44937a8bc57ae7`, `4129472442251f116892`, `99adabf9a305b184c711`, `14c764f4b6f8eb875c69`, `0ee617383119819f5e2f`, `7282db62f81adb1e42ef`, `42950f05d568d1a2f41c`, `e1f0bf855d278c91fca3`, `cd5c1a209edfd8b2ce62`.

Worst `mean.velocity_rel_l2` cases: `14c764f4b6f8eb875c69`, `42950f05d568d1a2f41c`, `a175caed84920dfa4fd0`, `82f3ebbc5d1bc6ba690e`, `7282db62f81adb1e42ef`, `c63b6a5ec7bf67782ba3`, `b9da34c40351567a09af`, `42967e2580a554437a66`, `94d4e097cc92f285dafa`, `de7313119fb7aa8191f5`, `11877b194ec023a42628`, `0c06caa66b36e84c55bb`, `5783775b16bb62f6cc19`, `e67eb3d529790f0ae9b8`, `90cc08bfc4228e7d31c1`, `f0263b072666f47e05aa`, `43fa9230ec19046ed59e`, `9de9327bd8db2edd1b85`, `64ce226eb4b419428b4f`, `c1606557cb50ca332105`.

Worst `nearest.cd_head_pct` cases: `b78611fddefe6b847391`, `b4eea8d300e034caeec6`, `0c06caa66b36e84c55bb`, `5783775b16bb62f6cc19`, `14c764f4b6f8eb875c69`, `f91adb7443b71acd6161`, `28e8b523ee6857d9acbb`, `4cdd44562b12b30ed918`, `1e35b68ef35aa51d2726`, `559c3836a6cf76af3140`, `4d63e335a3361b3074d7`, `11877b194ec023a42628`, `0f4be6158646be5d16e5`, `5a9e569cd1db7fb8d3c2`, `0224e55d2f834fcee90d`, `c63b6a5ec7bf67782ba3`, `b902f65ef22ffeae0d00`, `d880f18b19cfce4541a4`, `3c1a6e3dd4df7b6e23e4`, `f54d034f0d229245d17e`.

Worst `nearest.velocity_rel_l2` cases: `11877b194ec023a42628`, `b78611fddefe6b847391`, `14c764f4b6f8eb875c69`, `0c06caa66b36e84c55bb`, `5783775b16bb62f6cc19`, `b902f65ef22ffeae0d00`, `57431bcb961e8bf6d065`, `6381a94036198183d665`, `57bbc978da04dab2286e`, `c0999383edd5a57b44bb`, `b4eea8d300e034caeec6`, `61bcc957109b8c6d2757`, `6daf9d478a1fa11cf255`, `90cc08bfc4228e7d31c1`, `f91adb7443b71acd6161`, `b5a4a15f2ac67e90bbc0`, `85853b8052ba8f4bb274`, `0d4ef837234829654005`, `a48cd824680ce5660256`, `1040ba8f5757b76d3965`.

Worst `selected.cd_field_pct` cases: `68e1d1346fb8677b5a88`, `30ec572e5b27725a8ced`, `b3c127f27a0d2e03dad5`, `696afde58b9ef338be48`, `5913feb49d961a34ae44`, `961cccdade9d8c26dc82`, `38eab8bcb17fa29012fe`, `83e4502764acc64d986f`, `bbe20bbc8d50fbf047b3`, `8dd5d3745cfc35a2d296`, `4fd79e3bfbdba96acf0b`, `7fa88ebcf2f73860a08c`, `501a852a693218d51b7a`, `507045515c88609d0bc3`, `79fc12ee442394aeff2d`, `e4d75a42e710cb17662e`, `0399be54edc0599015ed`, `22589dbe3591d7c64398`, `9318c5eb142c0e89a2ae`, `15ddf7a52de44e09ef15`.

Worst `selected.cd_head_pct` cases: `11877b194ec023a42628`, `c63b6a5ec7bf67782ba3`, `0d4ef837234829654005`, `a48cd824680ce5660256`, `7fa88ebcf2f73860a08c`, `90cc08bfc4228e7d31c1`, `14c764f4b6f8eb875c69`, `101fbe5765b1f5afc638`, `696afde58b9ef338be48`, `8c6f5146fff25bb058bb`, `b902f65ef22ffeae0d00`, `961cccdade9d8c26dc82`, `6510c0695f315aec6361`, `1047e3b634ecc80cac94`, `68c3613c1fb8ccc5f951`, `68e1d1346fb8677b5a88`, `e0375780af4682fa1e9e`, `5a9e569cd1db7fb8d3c2`, `f0672fb1c166897d0ae6`, `b3c127f27a0d2e03dad5`.

Worst `selected.head_field_gap_pct` cases: `30ec572e5b27725a8ced`, `5913feb49d961a34ae44`, `68e1d1346fb8677b5a88`, `b3c127f27a0d2e03dad5`, `696afde58b9ef338be48`, `83e4502764acc64d986f`, `38eab8bcb17fa29012fe`, `8dd5d3745cfc35a2d296`, `4fd79e3bfbdba96acf0b`, `501a852a693218d51b7a`, `bbe20bbc8d50fbf047b3`, `79fc12ee442394aeff2d`, `507045515c88609d0bc3`, `0399be54edc0599015ed`, `15ddf7a52de44e09ef15`, `961cccdade9d8c26dc82`, `618d566beb447aeae575`, `e4d75a42e710cb17662e`, `9318c5eb142c0e89a2ae`, `b5fc59f1def3a17bd4db`.

Worst `selected.obstacle_ratio` cases: `cc9c915dbbd044f03dfa`, `b3c127f27a0d2e03dad5`, `b5a4a15f2ac67e90bbc0`, `c0999383edd5a57b44bb`, `5783775b16bb62f6cc19`, `aabf87cd9ccca41c72a3`, `80d666ac5f220b05bf9e`, `e4d75a42e710cb17662e`, `2d71dd421e0266036051`, `bebfe773546cc7fab89a`, `f3cf709cc38c5ebe052d`, `d880f18b19cfce4541a4`, `4a4096bd2c4c48a33d01`, `d22c7967183570cd9c29`, `a8e5ea6ab49b95adc036`, `43fa9230ec19046ed59e`, `f0672fb1c166897d0ae6`, `ad1043324db3b529133d`, `3947939e61676c2f4f71`, `26ccdcdeabb3d5cebbfc`.

Worst `selected.prediction_div_mean_abs` cases: `0399be54edc0599015ed`, `4fd79e3bfbdba96acf0b`, `14c764f4b6f8eb875c69`, `8c6f5146fff25bb058bb`, `8060b39a09726a2f790a`, `ecca1b66959ffc34be3d`, `42950f05d568d1a2f41c`, `507045515c88609d0bc3`, `6510c0695f315aec6361`, `b12971de8b3edcbf49b7`, `1ec2cbc77735b412a6da`, `3c1a6e3dd4df7b6e23e4`, `4a4096bd2c4c48a33d01`, `6e1439e1e78a7a209841`, `30ec572e5b27725a8ced`, `6f765738bf7f10d92d83`, `700670435e36171c4595`, `2c7c4f40dc63ef88d947`, `e45748da8603252d163a`, `68c3613c1fb8ccc5f951`.

Worst `selected.solver_div_mean_abs` cases: `4fd79e3bfbdba96acf0b`, `b3c127f27a0d2e03dad5`, `8dd5d3745cfc35a2d296`, `c0999383edd5a57b44bb`, `ecca1b66959ffc34be3d`, `38eab8bcb17fa29012fe`, `03d01690ab490aa5cda5`, `8060b39a09726a2f790a`, `15ddf7a52de44e09ef15`, `4d00679b63fcdd47e2eb`, `5a9e569cd1db7fb8d3c2`, `6510c0695f315aec6361`, `30ec572e5b27725a8ced`, `68c3613c1fb8ccc5f951`, `961cccdade9d8c26dc82`, `9318c5eb142c0e89a2ae`, `6e1439e1e78a7a209841`, `1040ba8f5757b76d3965`, `08ad285d5c10e1ec3e2b`, `83e4502764acc64d986f`.

Worst `selected.velocity_rel_l2` cases: `11877b194ec023a42628`, `c63b6a5ec7bf67782ba3`, `b902f65ef22ffeae0d00`, `c0999383edd5a57b44bb`, `0d4ef837234829654005`, `82f3ebbc5d1bc6ba690e`, `68c3613c1fb8ccc5f951`, `90cc08bfc4228e7d31c1`, `4fd79e3bfbdba96acf0b`, `b3c127f27a0d2e03dad5`, `6e1439e1e78a7a209841`, `14c764f4b6f8eb875c69`, `c67849f1baf50fb82999`, `1040ba8f5757b76d3965`, `03d01690ab490aa5cda5`, `42950f05d568d1a2f41c`, `ecca1b66959ffc34be3d`, `8efea3910d1dd420ca62`, `8060b39a09726a2f790a`, `6510c0695f315aec6361`.

## OOD heuristic

- Status: `valid`
- Median OOD normalized variance: `0.0085181`
- Median ID-boundary normalized variance: `0.00243732`
- OOD / ID-boundary ratio: `3.49486`

## Rotation sensitivity

Agreed signs: **4 of 10**.

| Case | Autograd Cd/degree | Central difference Cd/degree | Agrees |
|---|---:|---:|:---:|
| `8dd5d3745cfc35a2d296` | 0.00524455 | 0 | no |
| `e1f0bf855d278c91fca3` | 0.00268032 | 0 | no |
| `94d4e097cc92f285dafa` | 0.00511037 | 0 | no |
| `e45748da8603252d163a` | 0.0136668 | 0.015625 | yes |
| `5913feb49d961a34ae44` | 0.0111164 | 0.0078125 | yes |
| `329ecb192e0552186bca` | 0.00890409 | 0 | no |
| `26ccdcdeabb3d5cebbfc` | 0.00555689 | 0 | no |
| `1e6ee553668166b1c5d2` | 0.00937917 | 0.0078125 | yes |
| `f0263b072666f47e05aa` | 0.00247993 | 0 | no |
| `ad1043324db3b529133d` | 0.00508001 | 0.0078125 | yes |

## Diagnostic plots

- [Representative flow fields](validation.plots/representative-fields.svg)
- [Worst-case flow fields](validation.plots/worst-fields.svg)
- [Model and baseline comparison](validation.plots/baseline-comparison.svg)
- [Error by design parameter](validation.plots/error-by-design.svg)
- [Cd head and field consistency](validation.plots/head-vs-field.svg)
- [Divergence and obstacle compliance](validation.plots/divergence-compliance.svg)
- [OOD ensemble variance](validation.plots/ood-variance.svg)
- [Rotation sensitivity agreement](validation.plots/sensitivity.svg)

## Provenance

- Source revision: `b9f730af0bfd00a933e23709c5c2abab33b0bb05`
- Source dirty: `false`
- Lock SHA-256: `52798c0af1cd5e55ecde6e054af54415f5fb8aae0e63dbd2d4e7972d203386ab`
- Config SHA-256: `22aeb4cd1aefbf6ec972ec5ac928cdb391de881ab042b8a415c2cf2d42c9f1c6`
- Device class: `L40S`
- Report SHA-256: `4995ef8f8456030f467d63b06cb4db63c43d22988d3767367b6a10b14c4fb189`
