# TODO

## general/larger projects
- make a widget for visualizing data, likely with holoviews or bokeh
- choose what data to plot for all
- make choices about how to have plots
- figure out how to add panel labels to figures easily
- check if there are any special calls open at journals at the moment or in the near term that might be better visibility for us
- ~~update set_paths for all the new data~~

## coherence
- ~~add method for plotting standard bands~~
    - ~~still need to fix legend plotting, but pretty good~~
- LATER: add method for plotting flexible bands
- fix method for plotting

## reconstructions
- ~~compare optimal to none lambda penalties~~
- link rasters of actual and model here
- ~~add legend~~

## fig 5
- pick channels
- pick scales
- get nice plots of
    - coherence
    - reconstruction
    - coherence bands
    - rasters of actual vs. glm

## rasters
- ~~gather spike time data from biophysical models~~
- ~~gather code from poster figures~~
- ~~make methods for opening rasters from biophysical models and from glms~~
- ~~make methods for creating psth from rasters~~
- ~~make methods for plotting rasters~~
- ~~make methods for plotting psth with rasters~~
- ~~do comparison of biophysical model PSTH and GLM PSTH~~
- do comparison of biophysical model PSTH and GLM psth with mse or corr?

## example stack
- ~~gather data for model data stack as in Fig 3 from poster~~
- ~~gather code from making poster figures~~
- ~~make methods to make plot based on channel~~
- add channel to example stack

## biophysical modeling
- ~~gather data for model output, actual vm, spike times - will need to .gitignore data I think, way too large~~
- ~~gather code for poster figures~~
- analyze actual model structure to get better depiction of channel variants
- understand why Ca2+ channels are called differently
- order the channels more logically in plots
- plot channel distributions as in poster Fig. 1
- plot stimuli Vm responses as an example as in poster Fig. 2
- compute and plot PSD of stimuli
- figure out if I can recover the random seed the stimulus was obtained from?

## directories
- alon:     ~/ngglasgow@gmail.com/Data_Urban/NEURON/AlmogAndKorngreen2014/par_ModCell5_thrdsafe/
- bhalla:   ~/ngglasgow@gmail.com/Data_Urban/NEURON/Bhalla_par_scaling/