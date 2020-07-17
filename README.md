# biophys_glm
Combined biophysical and statistical modeling pipeline for investigating roles of ion channels in stimulus encoding

## Work in Progress
This project is a work in progress working towards draft submission. The jist of the project can be seen in the poster (/manuscript/sand2019_poster_b.pdf). Only a subset of data are available in this repo. 

### State of code
The code here was still in process of being made into a fully funtional API, with goal of generating suite of plots for any given model. The idea is to give the user the choice of which channel/s and scales to plot for easy comparisons. This wasn't necessary for submission, and so I have moved ahead with generating minimal plots for publication while working on the draft.

Generally speaking, the files that do not start with 'plot' will be the modules for it's description. The 'plot' files will do what the descriptor is trying to say. 

There is a lot of old code in /archive. This served as my trash bin, which I intend to clean later on, but didn't want to throw away anything hastily.

### Using the data
You should be able to use the path structure from wherever you put this project with an arg to set_paths.Paths. The dir structure of the repo is consistent across machines, and the only thing that should change is the `path_to_project` arg. You'll have to change `path_to_project` in all the 'plot*.py' files. I guess I should've made this an sys arg or env variable or something...

You'll want to run a find/replace for `projects/biophys_glm_show` to `my_path/biophys_glm_show` and it should work brilliantly.

There's a requirements.txt that is the base readout of my `conda list`. This is probably too specific, but easiest thing for me to do at the moment. You could build an env from it, or just refer to it if you have any errors.

All of the 'plot*.py' files should generate plots. 
