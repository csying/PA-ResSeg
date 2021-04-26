#!/usr/bin/python
import configparser

def load_train_ini(ini_file):
    # initialize
    cf = configparser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    param_sections = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(numChannels    = cf.getint(s[d], "numChannels"),
                          mode         = cf.get(s[d], "mode"),
                          ins    = cf.getint(s[d], "ins"),
                          ous   = cf.getint(s[d], "ous"),
                          interv    = cf.getint(s[d], "interv"),
                          nclass  = cf.getint(s[d], "nclass"),
                          batch_size    = cf.getint(s[d], "batch_size"),
                          result_dir = cf.get(s[d], "result_dir"),
                          folder = cf.get(s[d],"folder"),
                          imgdir = cf.get(s[d], "imgdir"),
                          testdir = cf.get(s[d], "testdir"),
                          pre_train = cf.get(s[d],"pre_train"),
                          ckpt_path = cf.get(s[d],"ckpt_path"),
                          model_loc = cf.get(s[d], "model_loc"),
                          epoch = cf.getint(s[d], "epoch"),
                          epoch_save = cf.getint(s[d],"epoch_save"),
                          model = cf.get(s[d],"model"))


        # add to list
        param_sections.append(level_dict)

    return param_sections