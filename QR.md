# Towards Lesion-Agnostic Estimation via Inference-Time Inpainting

**Kayra Özdemir**
BSc. Artificial Intelligence, Vrije Universiteit Amsterdam
Dept. of Radiotherapy, Amsterdam UMC (VUmc)

<p align="center">
  <img src="demonstration.gif" width="750">
</p>

---

## Resources

📄 **Thesis**

[View Thesis](./thesis.pdf)

🖼️ **Scientific Poster**

[View Poster](./poster.pdf)
[View Poster (Night Mode)](./poster-night.pdf)

💻 **Repository**

[Repository Home](./README.md)

📚 **Full Poster References**

[Poster References](./POSTER_REFERENCES.md)

---

## Project Summary

| Item                  | Value    |
| --------------------- | -------- |
| Duration              | 6 months |
| Healthy Subjects      | 547      |
| Tumor Subjects        | 1,790    |
| Brain Age Models      | 2        |
| Tumor Generators      | 3        |
| Inpainting Frameworks | 3        |
| Experiments           | 20       |

> **Finding 1:** Tumors systematically bias brain age predictions
> **Finding 2:** Current inpainting approaches do not reliably restore healthy anatomy.

---

## Abstract

Brain age prediction models are increasingly used as biomarkers of brain health. However, most existing models are trained predominantly on healthy MRI and may therefore behave unpredictably in pathological populations. This work investigates whether synthetic tumor insertion systematically biases brain age estimates and whether inference-time inpainting can restore predictions toward a healthy baseline. Across multiple brain age models, tumor generators, and inpainting frameworks, focal lesions were found to systematically perturb brain age predictions, while current inpainting approaches generally failed to reliably recover healthy baseline estimates.

---

## Future Directions

* Validation on real tumor cohorts
* Additional brain age architectures
* Multi-modal MRI
* Regional and voxel-level brain age models
* Lesion-aware learning frameworks

---

## Acknowledgements

**Supervisors**

* dr. Szabolcs Dávid
* dr. Aneta Lisowska

---

[LinkedIn 🔗](https://www.linkedin.com/in/kayraozdemir/)

Thank you for taking the time to view my work.
