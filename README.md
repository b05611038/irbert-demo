# irbert-demo
This demonstration is based on the IR-BERT project, a full spectroscopic foundation model under manuscript preparation.

⚠️ **Note:** This demo implementation excludes the processor module, which contains IR-BERT’s novel wavelength–signal fusion architecture. However, the full interface and training scaffolding are preserved to demonstrate the system's modularity and practical deployment.

---
## 🚀 Quick Usage Demo

This example illustrates how the IR-BERT model can handle spectra with different wavelength ranges across devices.  
(*Note: the model call is retained for demonstration; processor logic is redacted, so this script does not run end-to-end in the demo.*)

```python
>>> import torch
>>> from IRBert import IRBertForMultiTaskPrediction, IRBertConfig
>>> model = IRBertForMultiTaskPrediction(IRBertConfig()).from_pretrained('finetuned-irbert-base')

>>> spectra_device1 = torch.randn(2, 900)
>>> wavelength_device1 = torch.arange(700, 2500, 2).float()
>>> output = model(spectra_device1, wavelength_device1)
>>> output.last_hidden_state.shape
torch.Size([2, 925, 768])

>>> spectra_device2 = torch.randn(2, 481)
>>> wavelength_device2 = torch.arange(887, 2330, 3).float()
>>> output = model(spectra_device2, wavelength_device2)
>>> output.last_hidden_state.shape
torch.Size([2, 506, 768])
```

## 🔧 Pretraining Example
[`pretrain_IRBERT_via_MTPrediction_and_custom_dataset.py`](./pretrain_IRBERT_via_MaskedSM_and_custom_dataset.py)  
Demonstrates how to pretrain IR-BERT using masked spectrum modeling with synthetic data and a custom loss function.  

## 🔧 Finetuning Example
[`finetune_IRBERT_via_MTPrediction_and_custom_dataset.py`](./finetune_IRBERT_via_MTPrediction_and_custom_dataset.py)  
Demonstrates how to finetune IR-BERT on synthetic multi-task datasets using a custom loss, metric function, and model loading interface.

## 📄 Citation & Publication Status
IR-BERT is currently under review. Processor architecture and spectral tokenization details are withheld in this public demo.  
For academic collaboration or inquiries, please contact the author.  

