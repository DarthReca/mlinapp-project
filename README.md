
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="center">

<h3 align="center"><img src=titolo.PNG></h3>

[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]

  <p align="center">
    <br />
    <a href="https://github.com/MLinApp-polito/mla-prj-02-am1.git/doc"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repo contains the implementation of a project made during the course *Machine Learning in Applications* attended at Politecnico di Torino. 
The objective of this project is to create a filter for facial attribute editing. It contains the original baseline and the
other techniques exploited to finetune the network.

The final report can be found [here](doc/report.pdf).

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

- [Pytorch](https://pytorch.org/)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

To setup the environment run the following script:  
  ```bash
  python setup_env.py
  ```
Otherwise install manually the dependencies with `pip install -r requirements.txt` and download the dataset and the desired weights.

The dataset can be found in CelebA [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and the weights in the Pytorch
implementation of AttGAN [repo](https://github.com/elvisyjlin/AttGAN-PyTorch).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

To train the model simply run `train.py` with the desired arguments. The ones used in report are:
- *lambda_1* : the weight of reconstruction loss
- *epochs* : number of training epochs to run
- *batch_size* : the size of batch
- *indices_path* : the numpy file containing the indices of the images to use for training
- *experiment_name* : the CometML experiment name
- *target_attr* : the attribute to modify
- *dg_ratio* : how many discriminator steps we run for each generator step
- *freeze_layers* : how many low layers of discriminator to freeze
- *use_alternate_dataset* : use **Alternate** method as described in report
- *max_time* : timer for stopping training
- *upload_weights* : if present upload checkpoint to CometML

For more details use `python train.py --help` and refer to original implementation.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

We used third-party AttGAN implementation see [ATTGAN](licenses/ATTGAN) for more details.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributors

The project was built by Damiano Bonaccorsi, [Daniele Rege Cambrin](https://github.com/DarthReca), Giulia D’Ascenzi, [Patrizio de Girolamo](https://github.com/patriziodegirolamo).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- He, Zhenliang et al. "AttGAN: Facial Attribute Editing by Only Changing What You Want." (2017).
- Mo, Sangwoo et al. "Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs." (2020).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/DarthReca/mlinapp-project.svg?style=flat
[contributors-url]: https://github.com/DarthReca/mlinapp-project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/DarthReca/mlinapp-project.svg?style=flat
[license-url]: https://github.com/DarthReca/mlinapp-project/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

