# Libraries
from bokeh.io import output_notebook, show
from bokeh.plotting import figure,output_file, show
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
import os
import eclabfiles as ecf
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
import re
# ---------


# Folder --> list of full paths (no DS_Store) --> sorted by num for aesthetic
def unpackDir(folder):
    tempList = os.listdir(folder)
    outputList = [folder + '/' + file for file in tempList if 'DS_Store' not in file]
    return outputList


# Naming Convention --> experiment details (TODO: guessing sucks --> regex or experiment JSON/YAML)
def nameToDetails(fileName):
    scanRate = re.findall('[0-9]+', fileName)[0]
    # test scan rate? 


    for guess in ['GC','PT']:
        if guess in fileName:
            typeWorkingElectrode = guess
        else:
            typeWorkingElectrode = 'Unknown'

    for guess in ['Ferro','Ferri','KCL','H2S04']:
        if guess in fileName:
            typeElectrolyte = guess
        else:
            typeElectrolyte = 'Unknown'


    return [scanRate,typeWorkingElectrode,typeElectrolyte]


# Works for a range of desired CV plots
def generateBokehFigure(dataInput,**optionalParams):
    
    # Appendable Parameters
    defaultParams = {'usableCycle': 2 ,'colorList': ['red','green','blue','purple','pink','brown','black'],
                     'inputFig':None, 'specificLabels': False,'title':'Cyclic Voltammogram'}
    workingParams = { **defaultParams, **optionalParams}


    # Generate Figure or append existing
    if workingParams['inputFig'] == None:
        fig = figure(width=1000, height=500)
    else:
        fig = workingParams['inputFig']

    # Default --> CV labeled by scan rate: make modifiable later
    fig.xaxis.axis_label = "Voltage (V)"
    fig.yaxis.axis_label = "Current (mA)"
    

    # Can input single file or file array or folder name
    if isinstance(dataInput, str):
        if 'mpr' in dataInput:
            fileList = [dataInput]
        else:
            # Create list of full paths if you input a folder name
            fileList = unpackDir(dataInput)
    else:
        fileList = dataInput
    

    # Add E-I curve for each file specified
    loopCount = 0 
    for fileName in fileList:
        
        [scanRate,typeWorkingElectrode,typeElectrolyte]=nameToDetails(fileName)
            
        if workingParams['specificLabels'] == True:
            labelText = '{} mv/s, {} WE'.format(scanRate,typeWorkingElectrode)
        else:
            labelText = '{} mv/s'.format(scanRate)

        #-----
            
        # Convert mpr file --> CSV and clean up data
        ecf.to_csv(fileName, csv_fn="workingData.csv")
        relevantData = pd.read_csv("workingData.csv")
        relevantData = relevantData.dropna()
        
        if workingParams['usableCycle'] != 'All':
            relevantData = relevantData[relevantData['cycle number'] == workingParams['usableCycle']]

        fig.line(relevantData['Ewe'], relevantData['<I>'],
               line_color = workingParams['colorList'][loopCount],
               line_dash = 'solid',
               legend_label = labelText)
            
        loopCount += 1

    
    # Specify figure legend
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    fig.title.text = workingParams['title']
    fig.title.align = "center"
    fig.title.text_font_size = "25px"


    return fig



# Extract fit values systematically for stirred data 
def analyzeStirredData(inputFile,**optionalParams):
    
    # Adopt any input specs and override defaults [change anodic/cathodic region to automated]
    defaultKwargs = {'anodicLimitRange': [.95 , 1 ] , 'cathodicLimitRanged': [-1 , -.95 ],'usableCycle': 2}
    workingParams = { **defaultKwargs, **optionalParams}
    
    
    # Convert the input EC Lab file to a csv
    ecf.to_csv(inputFile, csv_fn="workingData.csv")
    relevantData = pd.read_csv("workingData.csv")
    relevantData = relevantData.dropna()
    if workingParams['usableCycle'] != 'All':
        relevantData = relevantData.where(relevantData['cycle number'] == workingParams['usableCycle'])

    
    # Calculate limiting currents 
    anodicRegionialData = relevantData[(workingParams['anodicLimitRange'][0]  <= relevantData['Ewe']) & (relevantData['Ewe'] <  workingParams['anodicLimitRange'][1])]
    anodicLimitingCurrent = np.mean(anodicRegionialData['<I>'].values)

    cathodicRegionialData = relevantData[(workingParams['cathodicLimitRanged'][0]  <= relevantData['Ewe']) & (relevantData['Ewe'] <  workingParams['cathodicLimitRanged'][1])]
    cathodicLimitingCurrent = np.mean(cathodicRegionialData['<I>'].values)
    

    # Get full range values
    currentArray = relevantData['<I>'].values
    voltageArray = relevantData['Ewe'].values
    
    
    # Abandoned this technique due to viable alternative of sorting > 0 before taking log. Compare results sometime
    
    #--------
#     #Taking logs of negative numbers fucks with scipy --> must adopt a fitting range safely between limiting currents
    

#     definedIncrement = .005
#     # clean up code to clarify algorithm 
#     adoptedCurrentRange = np.arange(cathodicLimitingCurrent + .002, anodicLimitingCurrent - .002, definedIncrement) #add increment to kwargs
#     translatedVoltageArray = np.copy(adoptedCurrentRange)
    
    
#     for index in range(0,len(translatedVoltageArray)):
#         incrementCurrentMin = adoptedCurrentRange[index] - definedIncrement
#         incrementCurrentMax = adoptedCurrentRange[index] + definedIncrement
#         incrementData = relevantData[(incrementCurrentMin  <= relevantData['<I>']) & (relevantData['<I>'] <  incrementCurrentMax)]
#         translatedVoltageArray[index] = np.median(incrementData['Ewe'].values)
    #---------
    
    iFuncArray = np.divide(cathodicLimitingCurrent - currentArray, currentArray - anodicLimitingCurrent)
    usableIndices = np.where(iFuncArray > 0)[0]
    currentRatioLog = np.log(iFuncArray[usableIndices])
    usableVoltage = voltageArray[usableIndices]
    
    
    # Fit in realm of +-3 current function ratio --> param?
    limitedRatioIndices = np.where(np.abs(currentRatioLog)<3)
    
    model = np.polyfit(currentRatioLog[limitedRatioIndices],usableVoltage[limitedRatioIndices], 1)
    #print(model)
    predictedVoltage = model[0] * currentRatioLog + model[1]
    
    fittedVals = dict()
    fittedVals['cathodicLimitingCurrent'] = cathodicLimitingCurrent
    fittedVals['anodicLimitingCurrent'] = anodicLimitingCurrent
    fittedVals['offset'] = model[0]
    fittedVals['slope'] = model[1]
    
    # Sorting redundant due to cycling effect (find elegant solution)
    sortIndex = np.argsort(currentArray[usableIndices])
    
    return [fittedVals,predictedVoltage[sortIndex],currentArray[usableIndices][sortIndex]]

    
    
# Just a helper function for bokeh stuff
def dashAdder(inputFig,xArr,yArr):
    
    inputFig.line(xArr, yArr,
       line_color = 'black',
       line_dash = 'dashed')
    
# Another helper function for bokeh stuff (TODO: generalize dashing annotations)
def addTypicalMarks(inputFig,results):
    vert1x = [results['Ep'],results['Ep']]
    vert1y = [results['Ip_lower'],results['Ip']]
 
    vert2x = [results['Em'],results['Em']]
    vert2y = [results['Im'],results['Im_upper']]
          
    vert3x = [results['Ea'],results['Ea']]
    vert3y = [results['Im_Down'],results['Im_Up']]
          
    horz1x = [results['I0_left'],results['I0_right']]
    horz1y = [0,0]


    horz2x = [0,1]
    horz2y = [results['Ilp'],results['Ilp']]

    horz3x = [-1,0]
    horz3y = [results['Ilm'],results['Ilm']]
    
        
    dashAdder(inputFig,vert1x,vert1y)
    dashAdder(inputFig,vert2x,vert2y)
    dashAdder(inputFig,vert3x,vert3y)
    #dashAdder(inputFig,horz1x,horz1y)
    #dashAdder(inputFig,horz2x,horz2y)
    #dashAdder(inputFig,horz3x,horz3y)


# Extract parameters from unstirred file
def analyzeUnstirredFile(inputFile,**optionalParams):
    
    # Adopt any input specs and override defaults [change anodic/cathodic region to automated]
    defaultKwargs = {'usableCycle': 2}
    workingParams = { **defaultKwargs, **optionalParams}
    
    
    # Convert the input EC Lab file to a csv
    ecf.to_csv(inputFile, csv_fn="workingData.csv")
    relevantData = pd.read_csv("workingData.csv")
    #relevantData = relevantData.dropna()
    #print(relevantData['Ewe'])
    if workingParams['usableCycle'] != 'All':
        relevantData = relevantData[relevantData['cycle number'] == workingParams['usableCycle']]

    
    voltageArray = relevantData['Ewe'].to_numpy()
    currentArray = relevantData['<I>'].to_numpy()
    #print(voltageArray)

    #----------
    fitValues = dict()
    plusPeakIndex = np.argmax(currentArray)
    minusPeakIndex = np.argmin(currentArray)
    #print(plusPeakIndex)

    # Get peaks
    fitValues['Ep'] = voltageArray[plusPeakIndex] 
    fitValues['Em'] = voltageArray[minusPeakIndex]
    
    fitValues['Ip'] = currentArray[plusPeakIndex]
    fitValues['Im'] = currentArray[minusPeakIndex]
    
    fitValues['Ea'] = (fitValues['Ep'] + fitValues['Em'])/2
    
    # get limiting currents 
    topVIndex = np.argmax(voltageArray)
    bottomVIndex = np.argmin(voltageArray)
    
    fitValues['Ilp'] = currentArray[topVIndex]
    fitValues['Ilm'] = currentArray[bottomVIndex]
    
    fitValues['Elp'] = voltageArray[topVIndex]
    fitValues['Elm'] = voltageArray[bottomVIndex]

    
    # Get other points (for nice graphing)
    nearIndicesPlus = np.where(np.abs(voltageArray - fitValues['Ep']) < .003)[0]
    fitValues['Ip_lower'] = np.min(currentArray[nearIndicesPlus])
    
    nearIndicesMinus = np.where(np.abs(voltageArray - fitValues['Em']) < .003)[0]
    fitValues['Im_upper'] = np.max(currentArray[nearIndicesMinus])
    
    currentZeroIndices = np.where(np.abs(currentArray) < .001)[0]
    fitValues['I0_right'] = np.max(voltageArray[currentZeroIndices])
    fitValues['I0_left'] = np.min(voltageArray[currentZeroIndices])

    voltAveIndices = np.where(np.abs(voltageArray - fitValues['Ea']) < .003)[0]
    fitValues['Im_Up'] = np.max(currentArray[voltAveIndices])
    fitValues['Im_Down'] = np.min(currentArray[voltAveIndices])
    
    
    #--------
    
    return fitValues

# Consider this demo citation code
# # citation = Label(x=70, y=70, x_units='screen', y_units='screen',
# #                  text='Collected by Luke C. 2016-04-01', render_mode='css',
# #                  border_line_color='black', border_line_alpha=1.0,
# #                  background_fill_color='white', background_fill_alpha=1.0)

# #fig.add_layout(citation)


# Analyze flat region for capacitance
def analyzeCapacitance(inputFile,**optionalParams):
    
    # Adopt any input specs and override defaults [change anodic/cathodic region to automated]
    defaultKwargs = {'usableCycle': 2,'defaultRange': [-.5,.5]}
    workingParams = { **defaultKwargs, **optionalParams}
    
    
    # Convert the input EC Lab file to a csv
    ecf.to_csv(inputFile, csv_fn="workingData.csv")
    relevantData = pd.read_csv("workingData.csv")
    relevantData = relevantData.dropna()

    if workingParams['usableCycle'] != 'All':
        relevantData = relevantData[relevantData['cycle number'] == workingParams['usableCycle']]

    
    voltageArray = relevantData['Ewe'].values
    currentArray = relevantData['<I>'].values
    
    # Consider df.query not np.where
    reducedIndices = np.where(np.logical_and(voltageArray < workingParams['defaultRange'][1], voltageArray > workingParams['defaultRange'][0]))[0]
    
    reducedCurrent = currentArray[reducedIndices]
    positiveIndices = np.where(reducedCurrent > 0)[0]
    capacitativeCurrent = np.median(reducedCurrent[positiveIndices])
    
    return capacitativeCurrent


if __name__ == "__main__":

    demoFile =  '/Users/ethanmuchnik/Desktop/project-3/Data/CAT-WE_KOH_AGCL-RE/CV_PlatedWE_PTCE_AGCLRE_100mvs_C01.mpr'
    fig = generateBokehFigure(demoFile)
    show(fig)

    # results = analyzeUnstirredFile(demoFile,usableCycle = 2)
    # fig = generateBokehFigure(demoFile,usableCycle = 2)

    # print(results)
    # labelSource = ColumnDataSource(data=dict(V=[results['Ep'],results['Em'],results['Ea']],
    #                                 I=[results['Ip']/2,results['Im']/2,0.005],
    #                                 names=['Epa', 'Epc','Ea']))

    # labels = LabelSet(x='V', y='I', text='names',x_offset=5, 
    #                   y_offset=5, source=labelSource, render_mode='canvas')

    # addTypicalMarks(fig,results)
    # fig.add_layout(labels)
    # show(fig)

