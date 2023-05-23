```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
cd ../..

mv DNABERT/src/transformers ./transformers
rm -r DNABERT
```

```shell
conda install -c conda-forge biopython
apt install genometools

pip install tabulate
```