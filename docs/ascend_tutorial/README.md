# Ascend Tutorial
## Introduction

Ascend supports verl usage and development. This guide explains how to run verl on Huawei Ascend NPUs.

Last updated: 05/14/2026.

## Directory layout

```
ascend_tutorial/
├── get_start/                     # Getting started
├── feature_support/               # Feature support
├── model_support/                 # Model support
├── dev_guide/                     # Development guides
├── faq/                           # Frequently asked questions
└── contribution_guide/            # Contribution guides
```
## News
- [verl-ascend-recipe repository](https://github.com/verl-project/verl-ascend-recipe) - Ascend recipes are now available
- [verl on Ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - The 2026Q2 roadmap has been published

## Getting started
- [Docker build guide](./get_start/dockerfile_build_guidance.rst) - Build and use Docker images for Ascend
- [Custom environment installation](./get_start/install_guidance.rst) - Install a custom verl environment on Ascend NPUs
- [Quick start](./get_start/quick_start.rst) - Start running verl on Ascend NPUs

## Feature support

- [verl feature support](./dev_guide/model_dev/parameter_and_metrics.md) - Supported verl features and parameters
- [NPU feature support](./feature_support/npu_advance_features.md) - Common NPU features and environment variables

## Model support

- [Model and algorithm support](./model_support/model_and_algorithm_support.md) - Supported models and algorithms
- [Best-practice examples](./model_support/examples) - Best practices and model deployment examples


## Development guides

- [Model development](./dev_guide/model_dev)
    - [Model migration](./dev_guide/model_dev/transfer_to_npu_guide.md) - Model migration guide
    - [Training parameters and metrics](./dev_guide/model_dev/parameter_and_metrics.md) - Training parameters and metrics
    - [Model evaluation](./dev_guide/model_dev/evaluation.md) - Model evaluation guide
- [Numerical accuracy debugging](./dev_guide/precision_analysis)
    - [Numerical accuracy analysis](./dev_guide/precision_analysis/precision_alignment_zh.md) - Numerical alignment guide
    - [Precision debugger](./dev_guide/precision_analysis/precision_debugger_zh.md) - Tools for investigating numerical accuracy issues
- [Performance tuning](./dev_guide/performance)
    - [Performance analysis](./dev_guide/performance/ascend_performance_analysis_guide.md) - Performance analysis guide
    - [Performance tuning](./dev_guide/performance/perf_tuning_on_ascend.rst) - Performance tuning guide
    - [Profiling collection](./dev_guide/performance/ascend_profiling_zh.rst) - Profiling tool guide


## Support and feedback

If you encounter problems, use the following support channels:

1. Read the [FAQ](./faq/faq.rst)
2. Open a GitHub issue
3. Contact Ascend technical support

## Contributing
- [Contributing to verl](../contributing) - verl contribution guide
- [Ascend CI guide](./contribution_guide/ascend_ci_guide_zh.rst) - CI configuration and testing on Ascend

## Related resources

- [Official verl documentation](https://verl.readthedocs.io/)
- [Ascend developer community](https://www.hiascend.com/)
