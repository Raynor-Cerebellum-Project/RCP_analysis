Installing the pipeline
=======================

Use **Anaconda Prompt** on Windows, and make sure CUDA is installed.

**Clone the repository**

.. code-block:: bash

   git clone https://github.com/Raynor-Cerebellum-Project/RCP_analysis.git
   cd RCP_analysis

**Create the environment**

.. code-block:: bash

   conda env create -f environment.yml -n pipeline
   conda activate pipeline
   python -m pip install --upgrade pip

**Install easyocr**

Install without dependencies so that pip does not touch the PyTorch build that
``environment.yml`` pinned:

.. code-block:: bash

   python -m pip install easyocr --no-deps

**Install RCP_analysis**

.. code-block:: bash

   pip install -e .
