<!-- # transformer_control -->
# Can Transformers Perform Low-Level Control In-Context?

Ebonye Smith, Aidan Andrews, Phoenix Wilson, Alex Frias, Ayush Pandey, Gireeja Ranade<br>
University of California Berkeley<br>
Univiersity of Illinois Urbana-Champaign<br>
University of Southern California<br>
University of California Merced

This repository contains the implementation of **low-level transformer controller framework** that does stabilizing control for cartpole system and an acrobot system via incontext learning.

#### **Setup**
This repository requires two conda environments. Create a conda environment using environment_test.yml. This conda environment will be used for generating data, training, and doing inference. Create a second conda environment using environment_gym.yml. This conda environment is used only for visualization purposes when running gym_acrobot_inference.py or gym_cartpole_inference.py.


#### **1. Creating a Dataset**

To generate a dataset, use the following command:

```bash
python generate_dataset_cartpole_ebonye.py --config conf/_XandYtest_cartpole_ebonye.yaml
```

---

#### **2. Training the Model**

To train the model, run:

```bash
python trainSequential_ebonye_cartpole_zerodyn.py --config conf/_XandYtest_cartpole_ebonye.yaml
```

