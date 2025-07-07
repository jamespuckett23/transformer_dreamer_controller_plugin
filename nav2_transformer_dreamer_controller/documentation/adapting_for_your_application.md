# Adapting to train your model with Nav2

- A significant portion of this plugin sets up an initial approach for training a model to interact with Nav2
- There may be future official methods, but this approach works to begin with
- To allow this for your own training models, look at the following scripts:
    - scripts/load_gz.py: if you want to train with a gazebo simulation world
    - scrits/train.py: general script to train with this method
    - scrpts/train_google_colab.ipynb: load this script into google colabs and then run each necessary section to set up your environment to train in the cloud

Once the model is sufficiantly trained, you can load the model into the src/trans_dreamer_controller.cpp by saving the model in models/trained_models/transformer_dreamer.pt

