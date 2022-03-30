## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/https://github.com/ai-proteomics/aiproteomics) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/https://github.com/ai-proteomics/aiproteomics)](https://github.com/https://github.com/ai-proteomics/aiproteomics) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-aiproteomics-00a3e3.svg)](https://www.research-software.nl/software/aiproteomics) [![workflow pypi badge](https://img.shields.io/pypi/v/aiproteomics.svg?colorB=blue)](https://pypi.python.org/project/aiproteomics/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=https://github.com/ai-proteomics_aiproteomics&metric=alert_status)](https://sonarcloud.io/dashboard?id=https://github.com/ai-proteomics_aiproteomics) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=https://github.com/ai-proteomics_aiproteomics&metric=coverage)](https://sonarcloud.io/dashboard?id=https://github.com/ai-proteomics_aiproteomics) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/aiproteomics/badge/?version=latest)](https://aiproteomics.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/build.yml/badge.svg)](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/https://github.com/ai-proteomics/aiproteomics/actions/workflows/markdown-link-check.yml) |

## `aiproteomics` python package
This package contains various tools, datasets and ML model implementations from the field of (phospho-)proteomics. It is intended to facilitate the testing and comparison of different neural network architectures and existing models, using the same datasets. Both retention time and fragmentation (MSMS) models are included.

Implementations of existing models from the literature are intended to be modifiable/extendable. For example, so that tests may be carried out with different peptide input lengths etc.

## Installation

To install aiproteomics from GitHub repository, do:

```console
git clone https://github.com/https://github.com/ai-proteomics/aiproteomics.git
cd aiproteomics
python3 -m pip install .
```

## Contributing

If you want to contribute to the development of aiproteomics,
have a look at the [contribution guidelines](CONTRIBUTING.md).
