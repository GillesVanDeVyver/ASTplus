# Detecting Machine Failures with Microphones  
**Using AI to Enable Proactive Maintenance in Industrial Machines**  

Industrial machines are expensive assets, and proactive maintenance can help reduce costly downtime and repairs. Using microphones to monitor machine health is an effective solution, as subtle sound changes can indicate underlying issues that visual inspections might miss. However, interpreting these sounds is challenging due to background noise, limited data, and variations across machines and environments.  

This project leverages AI—specifically transformer-based models—to detect anomalies in machine sound data, enabling more efficient and reliable predictive maintenance.

---

## The Challenge  

In real-world factory scenarios:  
- Sound recordings are only available when machines are functioning *correctly*.  
- Factories are noisy, introducing *background noise* that complicates analysis.  
- Machines vary across environments, requiring cross-domain generalization.  
- Data availability is *limited*, posing a risk of overfitting.  

This problem formed the basis of the **DCASE 2021 Challenge**, which served as the setting for this work during my master's thesis.  

---

## Our Solution  

At the time of the challenge, **Vision Transformers (ViT)** had emerged as a powerful deep learning architecture. While ViTs were originally designed for natural images, their **attention mechanism** also showed promise for sound data represented as **spectrograms**.  

### Why Vision Transformers?  
- Spectrograms capture the contribution of different **frequencies over time**.  
- The attention mechanism allows the model to focus on **important frequencies** and **time points** indicative of anomalies.  

### Addressing Limited Data  
Transformer models are typically **data-hungry**. To overcome the issue of limited data:  
1. We used a **pretrained ViT** model trained on both natural images and spectrograms to extract features.  
2. Only a **small part of the model** was fine-tuned on our specific dataset.  

This approach enabled us to utilize the transformer’s **feature extraction capabilities** while avoiding overfitting on the small amount of available training data.  

---

## Publication and Presentation  


The work was published as a proceeding:

[**Adapted Spectrogram Transformer for Unsupervised Cross-Domain Acoustic Anomaly Detection**](https://ieeexplore.ieee.org/abstract/document/9980266)



## Contact

Developer: <br />
[https://gillesvandevyver.com/](https://gillesvandevyver.com/)


Management: <br />
danny.hughes@kuleuven.be <br />
sam.michiels@kuleuven.be <br />
