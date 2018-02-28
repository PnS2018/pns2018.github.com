---
layout: default
---

# Setup your laptop for Scientific Computing with Python

## General Advice on Operating System

Generally, you should use either a modern Linux or `*nix` distribution.
For most users, we recommend [Ubuntu](https://www.ubuntu.com/).
If you have a Mac, `macOS` is a custom flavor of `*nix` OS.
Please do not try to install other system such as Ubuntu or Arch Linux
on a Mac if you don't know what you are doing.

You should avoid using Windows as much as possible.
In this module, we can only offer our limited help if you are using Windows.

## Which Python Distribution Should I Use?

If you are a newbie to Python, you will probably just go to the official Python website, download the installation file and install on your computer. In fact, this may be the worst thing you can do.

So, in most modern Linux and `*nix` operating system, the Python distribution is installed for managing some system services. The system-installed Python can be used by user for sure. However, because this distribution manages system services and needs writing privilege to some directories constantly, we strongly advice that you should not touch the system-installed Python for new user. Moreover, you could suffer a very big trouble when you are trying to install packages that
build from scratch.

Instead, what we need is a Python distribution that has its own environment and can be easily removed when we want to. So the answer is [Anaconda](https://anaconda.org/), a Python distribution that is built for Scientific Computing.

Anaconda delivers a custom Python distribution that includes over 1000 data science related packages, check [here](https://docs.anaconda.com/anaconda/packages/py2.7_linux-64). After install, all the files of Anaconda is in a folder. And if you messed something up, simply remove that folder and Install it again!

### Anaconda setup instructions

1. Open a terminal and download Anaconda

    ```bash
    $ wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -O anaconda.sh  # for Linux
    ```

    ```bash
    $ curl https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh -o anaconda.sh  # for macOS
    ```

2. Install Anaconda

    ```bash
    $ bash ./anaconda.sh
    ```

    Follow the instruction and make sure Anaconda is added in your bash configuration file such as `.bashrc` or `.zshrc`.

3. Close the current terminal and open another one (so that the bash configuration is loaded again). Type `python`, you should see something similar to this:

    ```
    Python 2.7.14 |Anaconda custom (64-bit)| (default, Nov  8 2017, 22:44:41)
    [GCC 7.2.0] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>>
    ```

__Remark__: For Windows OS, please read the installation instruction [here](https://www.anaconda.com/download/#windows). Or, remove your Windows and install Linux.

## Which Python Version Should I Use?

Python 2.7

## How to Install Other Python Packages?

Generally, with Anaconda, we have three ways of installing other Python packages.

1. Use `pip`. `pip` is the official Python packaging system that manages package installation, uninstallation, version control, custom package building, etc. You can install additional Python packages by

    ```bash
    $ pip install some-package-name
    ```

    If the package is available in [PyPi](https://pypi.org/), `pip` will automatically pull the software from the website and install it.

2. Use `conda`. Anaconda uses the packaging system `conda` to manage packages and libraries installation. At heart, `conda` does package, dependency and environment management for any language. `conda` can pull and install pre-built package from a specific server and resolve dependency accordingly.

    ```bash
    $ conda install some-package-name
    ```

3. Use `setup.py`. A decent Python library usually has a `setup.py` script. With this script, you can install the package via

    ```bash
    $ python setup.py install
    ```

## Do I Need Anything Else?

For a beginner who wants to learn Python for data science, No, you generally don't need to install additional softwares. Anaconda has taken care of software installation.

However, you should be able to familiar with the software packaging system on your computer. For Debian family (including Ubuntu), that is `apt-get` or `apt`. For macOS, that is either `MacPorts` or `homebrew`.

## IDE

There are many different IDEs for Python on the market. Here, we list some of
them that generally deliver fantastic coding experience.

### Say NO to Jupyter Notebook

If you have taken a Python class or Python 101, you have probably seen that
instructors like Jupyter Notebook as a default presentation tool.
They even walk you though the steps so that you are comfortable with
this kind of coding style.

However, we are strongly against this paradigm of coding.
It is true that the Jupyter Notebook offered a nice presentation style.
But it is totally not designed for managing and engineering serious projects.

Therefore, we advice that you simply quit Jupyter Notebook.

### What do you generally need from an IDE?

+ Code auto-completion: let's just face it, no one is gonna remember hundreds or even thousands of commands.
+ Code navigation: if you want to check some definition or file, you need to get there fast.
+ Running the code: it should be easy to run the code.
+ Debugging: if there is anything wrong with the code, you should be able to find out which line is doing things wrongly.
+ Source code version control: Make sure your code is maintained and updated in a timely and efficient manner.
+ Static code checker: modern software engineering promotes the idea of producing high-quality _human readable_ code. Over the years, people has transformed specific coding styles into static code checker where the checker evaluates your code formatting and correct obvious errors.

### Atom

[Atom](https://atom.io/) is a very popular text editor that is built and maintained mainly by GitHub. This editor is modern and naturally integrates some of best software engineering practices. You can find tons of additional packages that helps you configure the editor to a Python IDE.

### PyCharm

[PyCharm](https://www.jetbrains.com/pycharm/) is a new Python IDE that has a beautiful interface and integrates all the features you will need for developing a Python project.

### Eclipse+PyDev

If you are a [Eclipse](http://www.eclipse.org/) user, perhaps you would be very happy to know that there is a dedicate Eclipse plugin for Python. [PyDev](http://www.pydev.org/) takes advantages of the powerful Eclipse compiling and debugging framework.

### Vim

For experienced user, we recommend [Vim](https://www.vim.org/) or its community-driven fork [neovim](https://neovim.io/). By configuring Vim, you will be able to get a fast editor that has all the IDE features you want and being lightweight in some sense. Additionally, because Vim is a very flexible editor, you could do many things very efficient while other editors won't or can't do.
