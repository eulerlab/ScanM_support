# ----------------------------------------------------------------------------
# scanm_smh.py
# Class for ScanM header files (`.smh`)
#
# The MIT License (MIT)
# (c) Copyright 2022 Thomas Euler
#
# 2022-01-30, first implementation
# ----------------------------------------------------------------------------
import array
import struct
import numpy as np
from scanm_global import *

# ----------------------------------------------------------------------------
class SMH(object):
  """Loads an `.smh` ScanM header file"""

  def __init__(self):
    self._dPreHdrDict = {}
    self._kvPairDict = {}

  def load(self, fName, verbose=False):
    """ Load header file `fName`
    """
    errC = Err_Ok
    fPath = fName +"." +SCMIO_headerFileExtStr
    scm_log(f"Opening file `{fPath}`")

    try:
      # Load pre-header into a dict
      scm_log("Loading pre-header ...")
      self._dPreHdrDict = scm_load_pre_header(fPath)

      # Load key-value pairs
      scm_log("Loading parameters (key-value pairs) ...")
      kvl = []
      nkv = 0
      done = False

      # Read all lines from header file
      with open(fPath, "r+t") as f:
        # Jump the pre-header ...
        f.seek(SCMIO_preHeaderSize_bytes)
        while not(done):
          ln = f.readline()
          if not(done := len(ln) == 0):
            # Line with content
            # -> convert to normal string
            # (this is a bit complicated because of special characters, 
            #  such as `µ`)
            buf = ln.encode("utf-8")
            tmp = ""
            ich = -1
            for j in range((len(buf)-1)//2):
              if buf[ich+1] == 0:
                ich += 2
                tmp += chr(buf[ich])
              else:
                ich += 3
                tmp = tmp[:-1] +chr(buf[ich-2]) +chr(buf[ich])

            s = "".join(tmp).split("\n")[0]
            if len(s) > 0 and s[0] != "\x00":
              # String is not empty and nd of file not reached
              # -> store string in list
              if verbose:
                scm_log(f"-> {nkv:5} {s}")
              kvl.append(s)
              nkv += 1
      if nkv == 0:
        scm_log("ERROR: No parameter(s) found")
        return Err_SMH_NoParametersFound


      # Now parse the found key-value pairs
      scm_log(f"{nkv} key-value pair(s) found")
      for i,s in enumerate(kvl):
        # Get the different elements of the key-value pair
        tmp = s.split(SCMIO_typeKeySep)
        sty = tmp[0]
        tmp = tmp[1].split(SCMIO_keyValueSep)
        svr = tmp[0].strip()
        tmp = [s.strip() for s in tmp[1].split(SCMIO_entrySep) if len(s)> 0]
        tid = None
        nvl = 1
        v = tmp[0]

        # Extract value(s) depending on type
        if sty == SCMIO_stringStr:
          svl = v.split(SCMIO_subEntrySep)
          if len(svl) == 1:
            svl = svl[0]
          else:
            nvl = len(svl)
          tid = np.character
        elif sty == SCMIO_real32Str:
          svl = float(v)
          tid = np.float64
        elif sty in [SCMIO_uint32Str, SCMIO_uint64Str]:
          if not v.lower() == "nan":
            svl = int(v)
            tid = np.uint32 if sty == SCMIO_uint32Str else np.uint64

        # Add parsed parameter to the parameter dictionary
        self._kvPairDict.update({svr: [tid, nvl, svl]})
        if verbose:
          scm_log(f"-> {i:5} {s}")

      # Some parameters need to be processed
      stimBufLenList = []
      tarStimDurList = []
      realStimDurList = []
      
      nStimBuf = self.get(SCMIO_keys.NumberOfStimBufs)
      for iBuf in range(nStimBuf):
        sKey = SCMIO_key_StimBufLen_x.format(iBuf)
        v = self.get(sKey, remove=True)
        stimBufLenList.append(v)

        sKey = SCMIO_key_Ch_x_TargetedStimDur.format(iBuf)
        v = self.get(sKey, remove=True)
        tarStimDurList.append(v)
        
        sKey = SCMIO_key_AO_x_Ch_x_RealStimDur.format("A", iBuf)
        v = self.get(sKey, remove=True)
        realStimDurList.append(v)
        
      # Add the 3 lists to dict
      self._kvPairDict.update(
        {SCMIO_keys.StimBufLenList.value: 
         [np.uint32, len(stimBufLenList), np.array(stimBufLenList)]
        })  
      self._kvPairDict.update(
        {SCMIO_keys.TargetedStimDurList.value:
         [np.float64, len(tarStimDurList), np.array(tarStimDurList)]
        })  
      self._kvPairDict.update(
        {SCMIO_keys.RealStimDurList.value:
         [np.float64, len(tarStimDurList), np.array(realStimDurList)]
        })  
        
      # Retrieve stimulus buffer map  
      StimBufMapEntr = np.array([[0]*SCMIO_maxStimBufMapEntries]*SCMIO_maxStimChans)
      mask = self.get(SCMIO_keys.StimulusChannelMask)
      for iCh in range(SCMIO_maxStimChans):
        if mask & (1 << iCh): 
          for iEntr in range(self.get(SCMIO_keys.MaxStimBufMapLen)):
            sKey = SCMIO_key_Ch_x_StimBufMapEntr_y.format(iCh, iEntr)
            v = self.get(sKey, remove=True)
            StimBufMapEntr[iCh][iEntr] = v  
      self._kvPairDict.update(
        {SCMIO_keys.StimBufMapEntries.value:
         [np.uint64, np.shape(StimBufMapEntr), StimBufMapEntr]
        })  
        
      # Retrieve list of input channel pixel buffer lengths 
      # (number of pixel puffer is continous, NOT equal to the AI channel index!)
      nInCh = 0
      pixBufLenList = []
      mask = self.get(SCMIO_keys.InputChannelMask)
      for iInCh in range(SCMIO_maxInputChans):
        if mask & (1 << iInCh):
          sKey = SCMIO_key_InputCh_x_PixBufLen.format(nInCh)
          v = self.get(sKey, remove=True)
          pixBufLenList.append(v)
          nInCh += 1 
      self._kvPairDict.update(
        {SCMIO_keys.NumberOfInputChans.value:
         [np.uint32, 1, nInCh]
        })  
      self._kvPairDict.update(
        {SCMIO_keys.InChan_PixBufLenList.value:
         [np.uint32, len(pixBufLenList), np.array(pixBufLenList)]
        })  

      scm_log(f"{len(self._kvPairDict)} parameter(s) extracted")
      self.summary()
      scm_log("Done.")

    except:
      raise
      #errC = Err_FileNotFound

    return errC == 0
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  @property
  def GUID(self):
    return self._dPreHdrDict["GUID"]

  @property
  def scanMode(self):
    return self.get(SCMIO_keys.USER_ScanMode)
  @property
  def scanType(self):
    return self.get(SCMIO_keys.USER_ScanType)

  @property
  def pixSize_byte(self):
    return self.get(SCMIO_keys.PixelSizeInBytes)
  @property
  def pixDurTarget_us(self):
    return self.get(SCMIO_keys.TargetedPixDur)
  @property
  def pixDur_us(self):
    return self.get(SCMIO_keys.RealPixDur)

  @property
  def nFr(self):
    return self.get(SCMIO_keys.FrameCounter)
  
  @property
  def nPixBufPerFr(self):
    return self.get(SCMIO_keys.USER_divFrameBufReq)
  
  @property
  def dxFr_pix(self):
    return self.get(SCMIO_keys.USER_dxPix)
  @property
  def dyFr_pix(self):
    return self.get(SCMIO_keys.USER_dyPix)

  @property
  def dxOffs_pix(self):
    return self.get(SCMIO_keys.USER_nXPixLineOffs)
  @property
  def dxRetrace_pix(self):
    return self.get(SCMIO_keys.USER_nPixRetrace)

  @property
  def nStimBuf(self): 
    return self.get(SCMIO_keys.NumberOfStimBufs)
  @property
  def stimChMask(self): 
    return self.get(SCMIO_keys.StimulusChannelMask)
  
  @property
  def nInputCh(self): 
    return self.get(SCMIO_keys.NumberOfInputChans)
  @property
  def inputChMask(self): 
    return self.get(SCMIO_keys.InputChannelMask)

  @property
  def zoom(self): 
    return self.get(SCMIO_keys.USER_zoom)

  '''
  'StimBufLenList': [numpy.uint32, 3, array([5120, 5120, 5120])],
  'TargetedStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
  'RealStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
  'StimBufMapEntries': [numpy.uint64, (32, 128), array([[0, 0, 0, ..., 0, 0, 0], ... [0, 0, 0, ..., 0, 0, 0]])],
  'InChan_PixBufLenList': [numpy.uint32, 3, array([2560, 2560, 2560])]}
  'MinVoltsAO': [numpy.float64, 1, -4.0],
  'MaxVoltsAO': [numpy.float64, 1, 4.0],
  'MaxStimulusBufferMapLength': [numpy.uint32, 1, 1],
  'Oversampling_Factor': [numpy.uint32, 1, 25],
  'MinVoltsAI': [numpy.float64, 1, -1.0],
  'MaxVoltsAI': [numpy.float64, 1, 5.0],
  'ScanPathFunc': [numpy.character,   8,   ['XYScan2', '5120', '80', '64', '10', '6', '0', '1']],
  'NSubPixOversamp': [numpy.uint32, 1, 25],
  'Angle_deg': [numpy.float64, 1, 0.0],
  'IgorGUIVer': [numpy.character, 1, '0.0.38.02'],
  'XCoord_um': [numpy.float64, 1, -599.52],
  'YCoord_um': [numpy.float64, 1, 881.16],
  'ZCoord_um': [numpy.float64, 1, 6213.9],
  'ZStep_um': [numpy.float64, 1, 1.0],
  'NFrPerStep': [numpy.uint32, 1, 1],
  'XOffset_V': [numpy.float64, 1, 0.0],
  'YOffset_V': [numpy.float64, 1, 0.0],
  'dZPixels': [numpy.uint32, 1, 0],
  'ZPixRetraceLen': [numpy.uint32, 1, 0],
  'ZPixLineOffs': [numpy.uint32, 1, 0],
  'UsesZForFastScan': [numpy.uint32, 1, 0],
  'Comment': [numpy.character, 1, 'n/a'],
  'SetupID': [numpy.uint32, 1, 1],
  'LaserWavelength_nm': [numpy.uint32, 1, 0],
  'Objective': [numpy.character, 1, 'n/a'],
  'ZLensScaler': [numpy.uint32, 1, 1],
  'ZLensShifty': [None, 1, 1],
  'AspectRatioFrame': [numpy.float64, 1, 1.0],
  'UnusedValue': [numpy.uint32, 1, 6428160],
  'HeaderLengthInValuePairs': [numpy.uint64, 1, 71],
  'Header_length_in_bytes': [numpy.uint64, 1, 5362],
  '''
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  def summary(self):
    print(f"Summary")
    print(f"-------")
    print(f"Scan    : mode, type   : {self.scanMode}, {self.scanType}")
    print(f"Pixel   : size         : {self.pixSize_byte} bytes/pixel")
    print(f"          duration     : {self.pixDur_us} us ({self.pixDurTarget_us})")
    print(f"Frame   : x-y size     : {self.dxFr_pix} x {self.dyFr_pix} pixels")      
    print(f"          x-offset     : {self.dxOffs_pix} pixels")
    print(f"          x-retrace    : {self.dxRetrace_pix} pixels")
    print(f"          count        : {self.nFr} recorded")    
    print(f"          organisation : {self.nPixBufPerFr} pixel buffers/frame")
    print(f"Stimulus: # of buffers : {self.nStimBuf}")
    print(f"          mask         : {self.stimChMask:04b}")
    print(f"Input   : # of channels: {self.nInputCh}")
    print(f"          mask         : {self.inputChMask:04b}")
    print(f"Zoom factor            : {self.zoom:3.2f}")
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  def get(self, key, index=-1, remove=False):
    """ Get value for given key
    """
    key = key if isinstance(key, str) else key.value  
    val = self._kvPairDict[key]
    if index < 0:
      res = val[2]
    else:
      if isinstance(val[1], tuple):
        assert index >= 0 and index < val[1][0], "Index out of range"
        res = val[2][index]
      else:  
        assert index >= 0 and index < val[1], "Index out of range"
        res = val[2] if val[1] == 1 else val[2][index]
    if remove:
      _ = self._kvPairDict.pop(key)
    return res  
    
# ----------------------------------------------------------------------------
