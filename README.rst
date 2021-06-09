.. -*- coding: utf-8 -*-

.. image:: https://img.shields.io/badge/docs-stable-yellow.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.paper.ijcb2021_vision_transformer_pad/master/index.html
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.paper.ijcb2021_vision_transformer_pad/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_vision_transformer_pad/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_vision_transformer_pad/commits/master
.. image:: https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_vision_transformer_pad/badges/master/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_vision_transformer_pad/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_vision_transformer_pad
.. image:: https://img.shields.io/pypi/v/bob.paper.ijcb2021_vision_transformer_pad.svg
   :target: https://pypi.python.org/pypi/bob.paper.ijcb2021_vision_transformer_pad


============================================================================
On the Effectiveness of Vision Transformers for Zero-shot Face Anti-Spoofing
============================================================================

This package is part of the signal-processing and machine learning toolbox Bob_. 

This package contains source code to replicate the experimental results published in the following paper::

    @inproceedings{georgeijcb2021,
        author = {Anjith George and Sebastien Marcel},
        title = {On the Effectiveness of Vision Transformers for Zero-shot Face Anti-Spoofing},
        year = {2021},
        booktitle = {International Joint Conference on Biometrics (IJCB 2021)},
    }

If you use this package and/or its results, please consider citing the paper.

Installation
------------

The installation instructions are based on conda_ and works on **Linux systems
only**. `Install conda`_ before continuing.

Once you have installed conda_, download the source code of this paper and
unpack it.  Then, you can create a conda environment with the following
command::

    $ cd bob.paper.ijcb2021_vision_transformer_pad
    $ conda env create -f environment.yml
    $ conda activate bob.paper.ijcb2021_vision_transformer_pad  # activate the environment
    $ conda install gcc_linux-64
    $ buildout
    $ bin/bob_dbmanage.py all download --missing
    $ bin/train_generic.py --help  # test the installation

This will install all the required software to reproduce this paper.

Configuring the experiments
---------------------------

Once the environment is ready. The WMCA and HQWMCA databases need to be downloaded.
For downloading the WMCA database, please check the following link `Download Instructions for WMCA <https://www.idiap.ch/dataset/wmca>`_.
For downloading the HQ-WMCA database, please check the following link `Download Instructions for HQ-WMCA <https://www.idiap.ch/dataset/hq-wmca>`_.


After downloading the database, you need to set the paths to
those in the configuration files. Bob_ supports a configuration file
(``~/.bob_bio_databases.txt``) in your home directory to specify where the
databases are located. The absolute path to the database and the location store the annotations should be specified. 
Please specify the paths for the database like below::

    $ cat ~/.bob_bio_databases.txt
    [YOUR_WMCA_DB_DIRECTORY] = /databases/WMCA/
    [YOUR_WMCA_ANNOTATIONS_DIRECTORY] = /databases/WMCA/ANNOTATIONS/

Update
------

With the latest versions, the above part (specifying the location of DB) is handled by an rc file, in ``.bobrc`` file in the home folder::

    $ cat ~/.bobrc
    {
    "bob.db.batl.directory": "/databases/WMCA/",
    "bob.pad.face.database.batl.annotations_temp_dir": "/databases/WMCA/ANNOTATIONS/"
    "bob.paper.ijcb2021_vision_transformer_pad.preprocessed_directory:"/preprocessed/",
    "bob.paper.ijcb2021_vision_transformer_pad.cnn_output_dir":"/cnn_outputdir/"
    }
    
Similarly, set the paths for HQ-WMCA datasets as well.

Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
.. _bob package development: https://www.idiap.ch/software/bob/docs/bob/bob.extension/master/
.. _conda: https://conda.io
.. _install conda: https://conda.io/docs/install/quick.html#linux-miniconda-install
