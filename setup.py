from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A web-app tool that integrates imaging data with (spatial) transcriptomics data. This synergy unlocks a new field of radiotranscriptomics in order to create e CDSS that aids clinicians to diagnose patients based on imaging and omic data. Alongside deep learning, this tool also pretends to explore the usage of SHAP values alongside decision trees to improve the interpretability of clinical results. This integration with medical statistics also allows to stratify patients based on their risk of disease as well as treatment options for a given type of cancer and thus achieving authentic precision medicine.',
    author='Rafael Oliveira, PhD',
    license='MIT',
)
