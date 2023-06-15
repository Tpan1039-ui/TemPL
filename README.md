# [Model] TemPL 

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ai4protein/TemPL/"><img width="600px" height="auto" src="https://github.com/ai4protein/TemPL/blob/main/band.jpg"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json)
[![GitHub license](https://img.shields.io/github/license/ai4protein/TemPL)](https://github.com/ai4protein/TemPL/blob/main/LICENSE)

Updated on 2023.06.15



## Key Features

This repository provides the official implementation of TemPL (Temperature-Guided Protein Language Modeling).

Key features:
- OGT (Optimal Growth Temperature) prediction.
- Zero-shot mutant effect prediction.

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/SESNet/TemPL) 

## Details
TemPL is an innovative deep learning strategy for OGT (optimal growth temperature) prediction and zero-shot prediction of protein thermostability and activity, leveraging temperature-guided language modeling.



## Get Started

**Main Requirements**  
> biopython==1.81   
> numpy==1.24.3     
> pandas==2.0.2     
> scipy==1.10.1     
> tokenizers==0.13.3    
> torch==1.12.0     
> tqdm==4.65.0  
> transformers==4.30.2  

**Installation**
```bash
pip install -r requirements.txt
```

**Download Model**

[templ-base](https://drive.google.com/file/d/1sjl-0JNBr5EH5PXy6dbkcZaO50zYklGe/view)

[templ-fine-tuning-for-tm-datasets](https://drive.google.com/file/d/1jo3OMJSCNuB_To2gNjOSCqNVjmqo2dZI/view?usp=drive_link)


**Predicting OGT**
```bash
python predict_ogt.py --model_name templ-base \
--fasta ./datasets/OGT/ogt_small.fasta \
--output ogt_prediction.tsv
```


**Predicting Mutant Effect**

Using the templ-base model.
```shell
python predict_mutant.py --model_name templ-base \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```

Or using the templ that fine-tuned on the homologous sequence of the TM dataset.
```shell
python predict_mutant.py --model_name templ-tm-fine-tuning \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```


## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023templ,
      title={TemPL: A Novel Deep Learning Model for Zero-Shot Prediction of Protein Stability and Activity Based on Temperature-Guided Language Modeling}, 
      author={Pan Tan and Mingchen Li and Liang Zhang and Zhiqiang Hu and Liang Hong},
      year={2023},
      eprint={2304.03780},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
