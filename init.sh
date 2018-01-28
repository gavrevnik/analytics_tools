# brew update

# https://github.com/pyenv/pyenv
# brew install pyenv;
# pyenv install 3.x.x # python version;
# eval "$(pyenv init -)"


# brew install jupyter;
# pip3 install jupyter_contrib_nbextensions;
# jupyter contrib nbextension install --user;
# jupyter nbextension enable varInspector/main;
## extensions -> Skip-Traceback, Collapsible Headings



echo 'eval "$(pyenv init -)"' >> ~/.bash_profile;
echo 'export EDITOR="/usr/bin/nano"' >> ~/.bash_profile;
echo 'alias wj="jupyter notebook"' >> ~/.bash_profile;

# terminal git config
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'}
export PS1="\u@\h \[\e[32m\]\w \[\e[31m\]\$(parse_git_branch)\[\e[00m\]$ "


git config --global alias.hist log --pretty=format:'%h %ad | %s%d [%an]' --graph --date=short;
## запускать при обновлении файла gitignore_global
git config --global core.excludesFile '~/.gitignore_global'; 
echo '*.DS_Store' >> ~/.gitignore_global
echo '*.py[cod]' >> ~/.gitignore_global
echo 'catboost_info/' >> ~/.gitignore_global
echo '*/.ipynb_checkpoints/*.ipynb' >> ~/.gitignore_global



## working git with jpnb
pip3 install nbdime
nbdime config-git --enable --global

## alternative way to work with jpnb: https://www.youtube.com/watch?v=BEMP4xacrVc
pip install nbstripout nbconvert
# nbstripout --install # это уже в конкретном гите

# ssh-add ~/.ssh/rsa; # rsa for servers

# pip3 install seaborn;
# pip3 install scipy;
