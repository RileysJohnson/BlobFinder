#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3 // Use modern global access method and strict wave access.
	
// Copyright 2019 by The Curators of the University of Missouri, a public corporation //
//																										 //
// Hessian Blob Particle Detection Suite   //
//                                         //
// G.M. King Laboratory                    //
// University of Missouri-Columbia	   //
// Coded by: Brendan Marsh                 //
// Email: marshbp@stanford.edu		   //

// CONTENTS //

	//   I.  Main Functions
	//  II.  Scale-Space Functions
	// III.  Particle Measurements
	//  IV.  Preprocessing Functions
	//   V.  Utilities

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

						// MAIN FUNCTIONS //
						
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
 
// Detects Hessian blobs in a series of images in a chosen data folder.
// Be sure to highlight the data folder containing the images in the data browser before running.
Function/S BatchHessianBlobs()
	
	String ImagesDF=GetBrowserSelection(0)
	String CurrentDF=GetDataFolder(1)
	If( !DataFolderExists(ImagesDF) || CountObjects(ImagesDF,1)<1 )
		DoAlert 0,"Select the folder with your images in it in the data browser, then try again."
		Return ""
	EndIf
	
	// Declare algorithm parameters.
	Variable scaleStart					= 1	// In pixel units
	Variable layers	 					= 256
	Variable scaleFactor					= 1.5
	Variable detHResponseThresh			        = -2    // Use -1 for Otsu's method, -2 for interactive
	Variable particleType					= 1	// -1 for neg only, 1 for pos only, 0 for both
	Variable subPixelMult					= 1 	// 1 or more, should be integer.
	Variable allowOverlap					= 0
	
	// Retrieve parameters from user
	Prompt scaleStart,"Minimum Size in Pixels"
	Prompt layers,"Maximum Size in Pixels"
	Prompt scaleFactor,"Scaling Factor"
	Prompt detHResponseThresh,"Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)"
	Prompt particleType, "Particle Type (-1 for negative, +1 for positive, 0 for both)"
	Prompt subPixelMult, "Subpixel Ratio"
	Prompt allowOverlap, "Allow Hessian Blobs to Overlap? (1=yes 0=no)"
	DoPrompt "Hessian Blob Parameters", scaleStart,layers,scaleFactor,detHResponseThresh,particleType,subPixelMult,allowOverlap
	If( V_Flag == 1 )
		Return ""
	EndIf
	
	Variable minH = -inf, maxH = inf, minV = -inf, maxV = inf, minA = -inf, maxA = inf
	DoAlert 2, "Would you like to limit the analysis to particles of certain height, volume, or area?"
		If(V_flag==1)
			
			Prompt minH,"Minimum height"
			Prompt maxH,"Maximum height"
			Prompt minA,"Minimum area"
			Prompt maxA,"Maximum area"
			Prompt minV,"Minimum volume"
			Prompt maxV,"Maximum volume"
			DoPrompt "Constraints",minH,maxH,minA,maxA,minV,maxV
			If( V_flag == 1)
				Return ""
			EndIf
		
		EndIf
	
	// Make a Data Folder for the Series
	Variable NumImages=CountObjects(ImagesDF,1),i
	String SeriesDF=CurrentDF+UniqueName("Series_",11,0),TempName
	NewDataFolder/S $SeriesDF
	
	// Store the parameters being used	
	Make/N=13 /O Parameters
	Parameters[0]  = scaleStart
	Parameters[1]  = layers
	Parameters[2]  = scaleFactor
	Parameters[3]  = detHResponseThresh
	Parameters[4]  = particleType
	Parameters[5]  = subPixelMult
	Parameters[6]  = allowOverlap
	Parameters[7]  = minH
	Parameters[8]  = maxH
	Parameters[9]  = minA
	Parameters[10] = maxA
	Parameters[11] = minV
	Parameters[12] = maxV
	
	// Find particles in each image and collect measurements from each image.
	String imageDF
	Make/N=0 /O AllHeights, AllVolumes, AllAreas, AllAvgHeights
	For(i=0;i<numImages;i+=1)
		Wave im=WaveRefIndexedDFR($ImagesDF,i)
		Print "-------------------------------------------------------"
		Print "Analyzing image "+Num2Str(i+1)+" of "+Num2Str(numImages)
		Print "-------------------------------------------------------"
		
		// Run the Hessian blob algorithm and get the path to the image folder.
		imageDF = HessianBlobs(im,params=Parameters)
		
		// Get wave references to the measurement waves.
		Wave Heights = $(imageDF+":Heights")
		Wave AvgHeights = $(imageDF+":AvgHeights")
		Wave Areas = $(imageDF+":Areas")
		Wave Volumes = $(imageDF+":Volumes")
		
		// Concatenate the measurements into the master wave.
		Concatenate {Heights}, AllHeights
		Concatenate {AvgHeights}, AllAvgHeights
		Concatenate {Areas}, AllAreas
		Concatenate {Volumes}, AllVolumes
		
	EndFor
	
	// Determine the total number of particles.
	Variable numParticles = DimSize(AllHeights,0)
	Print "  Series complete. Total particles detected: ", numParticles
	
	SetDataFolder $CurrentDF
	Return SeriesDF
End

// Executes the Hessian blob algorithm on an image.
//		im : The path to the image to be analyzed.
//		params : An optional parameter wave with the 13 parameters to be passed in.
//					If the parameter wave is not present, the user will be prompted for them.
Function/S HessianBlobs(im,[params])
	Wave im, params
	
	// Declare algorithm parameters.
	Variable scaleStart					= 1	// In pixel units
	Variable layers	 						= Max( DimSize(im,0) , DimSize(im,1) ) /4
	Variable scaleFactor					= 1.5
	Variable detHResponseThresh			= -2 		// Use -1 for Otsu's method, -2 for interactive
	Variable particleType					= 1		// -1 for neg only, 1 for pos only, 0 for both
	Variable subPixelMult					= 1 		// 1 or more, should be integer.
	Variable allowOverlap					= 0
	
	// Decalare measurement ranges.
	Variable minH = -inf, maxH = inf, minV = -inf, maxV = inf, minA = -inf, maxA = inf
	
	// Retrieve parameters if given in the params wave, or prompt the user for them if not.
	If( ParamIsDefault(params) )
		
		Prompt scaleStart,"Minimum Size in Pixels" // Change wording?
		Prompt layers,"Maximum Size in Pixels"
		Prompt scaleFactor,"Scaling Factor"
		Prompt detHResponseThresh,"Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)"
		Prompt particleType, "Particle Type (-1 for negative, +1 for positive, 0 for both)"
		Prompt subPixelMult, "Subpixel Ratio"
		Prompt allowOverlap, "Allow Hessian Blobs to Overlap? (1=yes 0=no)"
		DoPrompt "Hessian Blob Parameters", scaleStart,layers,scaleFactor,detHResponseThresh,particleType,subPixelMult,allowOverlap
		If( V_Flag == 1 )
			Return ""
		EndIf
		
		DoAlert 2, "Would you like to limit the analysis to particles of certain height, volume, or area?"
		If(V_flag==1)
			
			Prompt minH,"Minimum height"
			Prompt maxH,"Maximum height"
			Prompt minA,"Minimum area"
			Prompt maxA,"Maximum area"
			Prompt minV,"Minimum volume"
			Prompt maxV,"Maximum volume"
			DoPrompt "Constraints",minH,maxH,minA,maxA,minV,maxV
			If( V_flag == 1)
				Return ""
			EndIf
		
		EndIf
		
	Else
		If(DimSize(params,0)<13)
			DoAlert 0,"Error: Provided parameter wave must contain the 14 parameters."
			Return ""
		EndIf
		
		scaleStart 					= params[0]
		layers 						= params[1]
		scaleFactor 					= params[2]
		detHResponseThresh			= params[3]
		particleType					= params[4]
		subPixelMult					= params[5]
		allowOverlap					= params[6]
		minH							= params[7]
		maxH							= params[8]
		minA							= params[9]
		maxA							= params[10]
		minV							= params[11]
		maxV							= params[12]	
	EndIf
	
	// Check parameters: Convert the scaleStart and scaleStop parameters from pixel units to scaled units squared.
	scaleStart 		= (scaleStart*DimDelta(im,0))^2 /2
	layers			= ceil( log( (layers*DimDelta(im,0))^2/(2*scaleStart))/log(scaleFactor) )
	subPixelMult	  	= Max(1,Round(subPixelMult))
	scaleFactor		= Max(1.1,scaleFactor)
	
	// Hard coded parameters.
	Variable gammaNorm = 1
	Variable maxCurvatureRatio = 10
	Variable allowBoundaryParticles = 1
	
	// Make a data folder for the particles.
	String CurrentDF = GetDataFolder(1)
	String NewDF = NameOfWave(im) +"_Particles"
	If(DataFolderExists(NewDF))
		NewDF = UniqueName(NewDF,11,2)
	EndIf
	NewDF = CurrentDF+NewDF
	NewDataFolder/S $NewDF
		
	// Store a copy of the original image. Only looking at the first layer right now.
	Make/N=(DimSize(im,0),DimSize(im,1)) /O Original
	Note Original, Note(im)
	SetScale/P x,DimOffset(im,0),DimDelta(im,0), Original
	SetScale/P y,DimOffset(im,1),DimDelta(im,1), Original
	Multithread Original = im[p][q][0]
	Wave im = Original
	
	// Declare needed variables.
	Variable numPotentialParticles,i,j,count=0,limP=DimSize(im,0),limQ=DimSize(im,1),padding,height,ii,jj
	
	// Calculate the discrete scale-space representation.
	Print "Calculating scale-space representation.."
	Wave L = ScaleSpaceRepresentation(im,layers,Sqrt(scaleStart)/DimDelta(im,0),scaleFactor)
	Rename L, ScaleSpaceRep
	Wave L = ScaleSpaceRep
	
	// Calculate gamma = 1 normalized scale-space derivatives
	Print "Calculating scale-space derivatives.."
	BlobDetectors(L,gammaNorm)
	Wave LG = :LapG
	Wave detH = :detH
	
	// If the user wants to, use Otsu's method for the blob strength threshold or find it interactively.
	If( detHResponseThresh == -1)
		Print "Calculating Otsu's Threshold.."
		detHResponseThresh = Sqrt(OtsuThreshold(detH,LG,particleType,maxCurvatureRatio))
		Print "Otsu's Threshold: "+Num2Str(detHResponseThresh)
	ElseIf( detHResponseThresh == -2 )
		detHResponseThresh = InteractiveThreshold(im,detH,LG,particleType,maxCurvatureRatio)
		Print "Chosen Det H Response Threshold: "+Num2Str(detHResponseThresh)
	EndIf

	// Detect particle candidates by identifying scale-space extrema.
	Print "Detecting Hessian blobs.."
	Make/N=0 /O mapMax,mapDetH,mapNum,Info
	FindHessianBlobs(im,detH,LG,detHResponseThresh,mapNum,mapDetH,mapMax,Info,particleType,maxCurvatureRatio)
	numPotentialParticles = DimSize(Info,0)
	
	// Info Wave Key:
		// Info[i][0]  = P Seed, central x-position of blob in pixel units.
		// Info[i][1]  = Q Seed, central y-position of blob in pixel units.
		// Info[i][2]  = NumPixels, number of pixels contained within particle.
		// Info[i][3]  = Maximum blob strength displayed by particle.
		// Info[i][4]  = pStart, left x-position of bounding box in pixel units.
		// Info[i][5]  = pStop, right x-position of bounding box in pixel units.
		// Info[i][6]  = qStart, bottom y-position of bounding box in pixel units.
		// Info[i][7]  = qStop, top x-position of bounding box in pixel units.
		// Info[i][8]  = scale, the scale at which the scale-space extrema was located.
		// Info[i][9]  = layer in the discrete scale-space representation.
		// Info[i][10] = 1 for maximal over scales and overlapped space, 0 else.
		// Info[i][11] = Number of parent blob, own number if maximal.
		// Info[i][12] = Number of blobs contained in support if maximal, else 0.
		// Info[i][13] = Unused.
		// Info[i][14] = -1 for particle rejected, 0 for undetermined, >0 for particle accepted with given particle number.
	
	// Remove overlapping Hessian blobs if asked to do so, or else allow nested particles.
	If( allowOverlap==0 )
		Print "Determining scale-maximal particles.."
		MaximalBlobs(info,mapNum)
	Else
		info[][10]=1
	EndIf
	
	// Initialize particle containment and acceptance status as undetermined.
	If(numPotentialParticles>0)
		Info[][13] = 0
		Info[][14] = 0
	EndIf
	
	// Make waves for the particle measurements.
	Make/O /N=(numPotentialParticles) Volumes
	Make/O /N=(numPotentialParticles) Heights
	Make/O /N=(numPotentialParticles,2) COM
	Make/O /N=(numPotentialParticles) Areas
	Make/O /N=(numPotentialParticles) AvgHeights
	
	// Make waves for subpixel particle position calculation.
	Make/N=(2,2) Hessian
	Make/N=(2,1) Jacobian
	Variable subPixX,subPixY
	
	// Variables for the particle measurement calculations.
	Variable/C centerOfMass
	Variable particleArea, particlePerim, bg, avgHeight, vol
	
	Print "Cropping and measuring particles.." 
	Variable Xpixels,Ypixels,subLimP,subLimQ,p0,q0,r0,x0,xf,y0,yf,dx=DimDelta(im,0),dy=DimDelta(im,1),subp0,subq0,seedLG
	For(i=numPotentialParticles-1;i>=0;i-=1)
	
		// If asked to do so, only consider non-overlapping particles.
		If( allowOverlap==0 && info[i][10]==0 )
			Continue
		EndIf
	
		// Make various cuts to eliminate bad particles less than one pixel.
		If( info[i][2] < 1 || (info[i][5]-info[i][4])<0 || (info[i][7]-info[i][6])<0 )
			Continue
		EndIf
		
		// Consider boundary particles?
		If( allowBoundaryParticles==0 && (info[i][4]<=2 || info[i][5]>=DimSize(im,0)-3 || info[i][6]<=2 || info[i][7]>=DimSize(im,1)-3) )
			Continue
		EndIf
		
		// Make a crop, mask, and perimeter image for the individual particle.
		padding = Ceil(max( info[i][5]-info[i][4]+2, info[i][7]-info[i][6]+2 ))
		Duplicate/R=[max(info[i][4]-padding,0),min(info[i][5]+padding,limP-1)][max(info[i][6]-padding,0),min(info[i][7]+padding,limQ-1)] /O im,$("Particle_"+Num2Str(count)),$("Mask_"+Num2Str(count)),$("Perimeter_"+Num2Str(count))
		Wave particle = $("Particle_"+Num2Str(count))
		Wave mask = $("Mask_"+Num2Str(count))
		Multithread mask = SelectNumber(mapNum[ScaleToIndex(mapNum,x,0)][ScaleToIndex(mapNum,y,1)][info[i][9]]==i,0,1)
		Wave perim = $("Perimeter_"+Num2Str(count))
		Multithread perim =  mask==1 && (p>0 && q>0 && p<(DimSize(perim,0)-1) && q<(DimSize(perim,1)-1)) && (mask[p+1][q]==0 || mask[p-1][q]==0 || mask[p][q+1]==0 || mask[p][q-1]==0 || mask[p+1][q+1]==0 || mask[p-1][q+1]==0 || mask[p+1][q-1]==0 || mask[p-1][q-1]==0) 
		
		// Make local crop of the pixel-resolution determinant of Hessian and Laplacian of Gaussian for interpolation.
		String name = "detH_"+Num2Str(count)
		Duplicate/O /R=[max(info[i][4]-padding,0),min(info[i][5]+padding,limP-1)][max(info[i][6]-padding,0),min(info[i][7]+padding,limQ-1)][info[i][9]] detH, $name
		Wave ParticleDetH = $name
		Duplicate/R=[max(info[i][4]-padding,0),min(info[i][5]+padding,limP-1)][max(info[i][6]-padding,0),min(info[i][7]+padding,limQ-1)][info[i][9]] /O LG, $("LG_"+Num2Str(count))
		Wave ParticleLG = $("LG_"+Num2Str(count))
		
		// Determine number of pixels in subpixel image and initialize subpixel particle, determinant of Hessian, and Laplacian of Gaussian.
		xPixels = Round(DimSize(mask,0)*SubPixelMult)
		yPixels = Round(DimSize(mask,1)*SubPixelMult)
		name = "SubPixDetH_"+Num2Str(count)
		Make/N=(xPixels,yPixels) $name
		Wave SubPixDetH = $name
		SetScale/P x,DimOffset(mask,0)-DimDelta(mask,0)/2+DimDelta(mask,0)/(2*SubPixelMult)  ,DimDelta(mask,0)/SubPixelMult,SubPixDetH
		SetScale/P y,DimOffset(mask,1)-DimDelta(mask,1)/2+DimDelta(mask,1)/(2*SubPixelMult)  ,DimDelta(mask,1)/SubPixelMult,SubPixDetH
		SetScale/P z,DimOffset(L,2)*DimDelta(L,2)^Info[i][9],DimDelta(L,2),SubPixDetH
		name = "SubPixLG_"+Num2Str(count)
		Duplicate/O SubPixDetH, $name
		Wave SubPixLG = $name
		name = "SubPixParticle_"+Num2Str(count)
		Duplicate/O SubPixDetH, $name
		Wave SubPixParticle = $name
		
		// Bilinear interpolation of the particle, determinant of Hessian, and Laplcian of Gaussian.
		Duplicate/O Perim, Expanded
		ExpandBoundary8(Expanded)
		Multithread SubPixDetH = (Expanded(x)(y)==1) ? BilinearInterpolate(ParticleDetH,x,y) : (Mask(x)(y) ? ParticleDetH(x)(y) : -1)
		Multithread SubPixLG = (Expanded(x)(y)==1) ? BilinearInterpolate(ParticleLG,x,y) : (Mask(x)(y) ? ParticleLG(x)(y) : -1)
		KillWaves/Z Expanded
		Multithread SubPixParticle = BilinearInterpolate(Particle,x,y)
		
		// Compute subpixel mask and perimeter images.
		name = "SubPixMask_"+Num2Str(count)
		Duplicate/O SubPixDetH, $name
		Wave SubPixMask = $name
		Multithread SubPixMask = 0
		subp0 = ScaleToIndex(SubPixMask,IndexToScale(im, info[i][0],0),0)
		subq0 = ScaleToIndex(SubPixMask,IndexToScale(im, info[i][1],1),1)
		ScanlineFill8_LG(SubPixDetH,SubPixMask,SubPixLG,subp0,subq0,0,fillVal=1)
		
		// Compute sub-pixel edge (perimeter) image.
		name = "SubPixEdges_"+Num2Str(count)
		Duplicate/O SubPixDetH, $name
		Wave SubPixEdgesDetH = $name
		Multithread SubPixEdgesDetH = SubPixMask && (SubPixMask[Min(p+1,xPixels-1)][q]==0 || SubPixMask[Max(0,p-1)][q]==0 || SubPixMask[p][Min(q+1,yPixels-1)]==0 || SubPixMask[p][Max(0,q-1)]==0)

		// Eliminate any single pixel width bridges and edges.
		Multithread SubPixMask = 0
		ScanlineFillEqual(SubPixEdgesDetH,SubPixMask,subp0,subq0,fillVal=1)
		ExpandBoundary4(SubPixMask)
		Multithread SubPixEdgesDetH = SubPixMask && (SubPixMask[Min(p+1,xPixels-1)][q]==0 || SubPixMask[Max(0,p-1)][q]==0 || SubPixMask[p][Min(q+1,yPixels-1)]==0 || SubPixMask[p][Max(0,q-1)]==0)

		// Find subpixel scale-space extrema centers. Estimate derivatives using central differences.
		p0 = info[i][0]
		q0 = info[i][1]
		r0 = info[i][9]
		Jacobian[0][0] = (detH[p0+1][q0][r0]-detH[p0-1][q0][r0])/2
		Jacobian[1][0] = (detH[p0][q0+1][r0]-detH[p0][q0-1][r0])/2
		Hessian[0][0] = detH[p0-1][q0][r0] -2*detH[p0][q0][r0] +detH[p0+1][q0][r0]
		Hessian[1][1] = detH[p0][q0-1][r0] -2*detH[p0][q0][r0] +detH[p0][q0+1][r0]
		Hessian[0][1] = ( detH[p0+1][q0+1][r0] + detH[p0-1][q0-1][r0] - detH[p0+1][q0-1][r0] - detH[p0-1][q0+1][r0] )/4
		Hessian[1][0] = Hessian[0][1]
		MatrixOp/O SubPixOffset = -inv(Hessian) x Jacobian
		subPixX = DimOffset(im,0)+DimDelta(im,0)*(info[i][0]+SubPixOffset[0])
		subPixY = DimOffset(im,1)+DimDelta(im,1)*(info[i][1]+SubPixOffset[1])
		
		// Calculate metrics associated with the particle.	
		bg = M_MinBoundary(subPixParticle,subPixMask)
		particle -= bg
		subPixParticle -= bg
		height = M_Height(particle,mask,0)
		vol = M_Volume(subPixParticle,subPixMask,0)
		centerOfMass = M_CenterOfMass(subPixParticle,subPixMask,0)
		particleArea = M_Area(subPixMask)
		particlePerim = M_Perimeter(subPixMask)
		avgHeight = vol / particleArea
		
		// Check if the particle is in range
		If( !(height>minH && height<maxH && particleArea>minA && particleArea<maxA && vol>minV && vol<maxV) )
			KillWaves/Z particle,mask,perim,ParticleLG
			KillWaves/Z SubPixDetH,SubPixEdgesDetH,ParticleDetH,SubPixMask,SubPixLG,SubPixParticle
			Continue
		EndIf
		
		// Accept the particle.
		Info[i][14] = count
		
		// Document the metrics in the wave note of each particle.
		Note/K particle
		Note particle,"Parent:"+NameOfWave(im)
		Note particle,"Date:"+Date()
		Note particle,"Height:"+Num2Str(height)
		Note particle,"Avg Height:"+Num2Str(avgHeight) 
		Note particle,"Volume:"+Num2Str(vol)
		Note particle,"Area:"+Num2Str(particleArea)
		Note particle,"Perimeter:"+Num2Str(particlePerim)
		Note particle,"Scale:"+Num2Str(Info[i][8])
		Note particle,"xCOM:"+Num2Str(Real(centerOfMass))
		Note particle,"yCOM:"+Num2Str(Imag(centerOfMass))
		Note particle,"pSeed:"+Num2Str(Info[i][0])
		Note particle,"qSeed:"+Num2Str(Info[i][1])
		Note particle,"rSeed:"+Num2Str(Info[i][9])
		Note particle,"subPixelXOffset:"+Num2Str(SubPixOffset[0])
		Note particle,"subPixelYOffset:"+Num2Str(SubPixOffset[1])
		Note particle,"subPixelXCenter:"+Num2Str(subPixX)
		Note particle,"subPixelYCenter:"+Num2Str(subPixY)
		
		// Make a folder for the particle and move it there
		name = "Particle_"+Num2Str(count)
		NewDataFolder/S $name
		MoveWave particle, :
		MoveWave mask, :
		MoveWave perim, :
		MoveWave ParticleLG, :
		MoveWave SubPixDetH, :
		MoveWave SubPixEdgesDetH, :
		MoveWave ParticleDetH, :
		MoveWave SubPixMask, :
		MoveWave SubPixLG, :
		MoveWave SubPixParticle, :
		
		// Store the metrics
		Volumes[count] 		= vol
		Heights[count] 		= height
		COM[count][0]  		= Real(centerOfMass)
		COM[count][1]  		= Imag(centerOfMass) 
		Areas[count] 		= particleArea
		AvgHeights[count]	= avgHeight
		
		// Display the particles in an image.
		If(count==0)
			NewImage im
			SetScale/P x,DimOffset(im,0),DimDelta(im,0),"m",im
			SetScale/P y,DimOffset(im,1),DimDelta(im,1),"m",im
			Label left " "
			Label top " "
		EndIf
		AppendImage/T SubPixMask
		ModifyImage $NameOfWave(SubPixMask) explicit=1,eval={1,65535,0,0,15008},eval={0,-1,-1,-1}
		AppendImage/T SubPixEdgesDetH
		ModifyImage $NameOfWave(SubPixEdgesDetH) explicit=1,eval={1,65535,0,0},eval={0,-1,-1,-1}
		
		// If you want to see the fun in action, uncomment below.
		//DoUpdate
		
		SetDataFolder $NewDF
		count += 1
		
	EndFor
	
	// Make a map image showing where particles are found.
	Duplicate/O im, ParticleMap
	ParticleMap = -1
	For(i=count-1;i>=0;i-=1)
		
		name = "Particle_"+Num2Str(i)
		SetDataFolder $name
		name = "Mask_"+Num2Str(i)
		Wave Mask = $name
		For(ii=0;ii<DimSize(Mask,0);ii+=1)
		For(jj=0;jj<DimSize(Mask,1);jj+=1)
			If( Mask[ii][jj] )
				ParticleMap[ScaleToIndex(ParticleMap,IndexToScale(Mask,ii,0),0)][ScaleToIndex(ParticleMap,IndexToScale(Mask,jj,1),1)]=i
			EndIf
		EndFor
		EndFor
		
		SetDataFolder $NewDF
	EndFor
	
	// Trim the metric waves of excess points.
	DeletePoints/M=0 count,numPotentialParticles,Volumes,Heights,COM,Areas
	
	// Kill leftover waves.
	KillWaves/Z HeightWindow, HeightCoefs, :W_ParamConfidenceInterval, :W_sigma, Hessian, Jacobian
	KillWaves/Z :SubPixOffset,:SS_MAXSCALEMAP,:detH_MaxValues,mapMax,mapNum,mapDetH
	
	// Return to the orginal data folder.
	SetDataFolder $CurrentDF
	
	Return NewDF
End

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

						// SCALE-SPACE FUNCTIONS //
						
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

// Computes the discrete scale-space representation L of an image.
//		im : The image to compute L from.
//		layers : The number of layers of L.
//		t0 : The scale of the first layer of L, provided in pixel units.
//		tFactor : The scaling factor for the scale between layers of L.  
Function/Wave ScaleSpaceRepresentation(im,layers,t0,tFactor)
	Wave im
	Variable layers, t0, tFactor
	
	// Convert t0 to image units.
	t0 = (t0*DimDelta(im,0))^2
	
	// Go to Fourier space.
	FFT im
	
	// Make the layers of the scale-space representation and convolve in Fourier space.
	Variable i
	String names=""
	Make/N=(layers) /WAVE /O Family
	For(i=0;i<layers;i+=1)
		Make/N=(DimSize(im,0),DimSize(im,1)) /C /O $("L_"+Num2Str(i))
		Family[i] = $("L_"+Num2Str(i))
		SetScale/P x,DimOffset(im,0),DimDelta(im,0),Family[i]
		SetScale/P y,DimOffset(im,1),DimDelta(im,1),Family[i]
		Wave/C Layer = Family[i]
		Multithread Layer = im[p][q]*exp( -(x^2+y^2)*pi^2*2*(t0*(tFactor^i)) )
		IFFT Layer
		names += "L_"+Num2Str(i)+";"
	EndFor

	// Return to spatial domain.
	IFFT im
	
	// Concatenate layers of the scale-space representation into a 3D wave.
	Concatenate/Kill /O names, $(NameOfWave(im)+"_L")
	Wave L = $(NameOfWave(im)+"_L")
	SetScale/P x,DimOffset(im,0),DimDelta(im,0),L
	SetScale/P y,DimOffset(im,1),DimDelta(im,1),L
	SetScale/P z,t0,tFactor,L
	
	//KillWaves/Z Family
	
	Return L
End

// Computes the two blob detectors, the determinant of the Hessian and the Laplacian of Gaussian.
//		L : The scale-space representation of the image.
//		gammaNorm : The gamma normalization factor, see Lindeberg 1998. Should be set to 1 in most blob detection cases.
Function BlobDetectors(L,gammaNorm)
	Wave L
	Variable gammaNorm
	
	// Make convolution kernels for calculating central difference derivatives.
	Make/N=(5,1) /O LxxKernel = {{0,0,0,0,0},{0,0,0,0,0},{-1/12,16/12,-30/12,16/12,-1/12},{0,0,0,0,0},{0,0,0,0,0}}
	Make/N=(1,5) /O LyyKernel = {{0,0,-1/12,0,0},{0,0,16/12,0,0},{0,0,-30/12,0,0},{0,0,16/12,0,0},{0,0,-1/12,0,0}}
	Make/N=(5,5) /O LxyKernel = {{-1/144,1/18,0,-1/18,1/144},{1/18,-4/9,0,4/9,-1/18},{0,0,0,0,0},{-1/18,4/9,0,-4/9,1/18},{1/144,-1/18,0,1/18,-1/144}}
	
	// Compute Lxx and Lyy. (Second partial derivatives of L).
	MatrixOp/O /NPRM Lxx = convolve(L,LxxKernel,-2)
	MatrixOp/O /NPRM Lyy = convolve(L,LyyKernel,-2)
	
	// Compute the Laplacian of Gaussian.
	MatrixOp/O /NPRM LapG = Lxx + Lyy
	
	// Set the image scale.
	SetScale/P x,DimOffset(L,0),DimDelta(L,0),LapG
	SetScale/P y,DimOffset(L,1),DimDelta(L,1),LapG
	SetScale/P z,DimOffset(L,2),DimDelta(L,2),LapG

	// Gamma normalize and account for pixel spacing.
	Multithread LapG *= (DimOffset(L,2)*DimDelta(L,2)^r)^(gammaNorm) / (DimDelta(L,0)*DimDelta(L,1))
	
	// Fix errors on the boundary of the image.
	FixBoundaries(LapG)
	
	// Compute the determinant of the Hessian.
	MatrixOp/O /NPRM detH = fp32(Lxx * Lyy - powr(convolve(L,LxyKernel,-2),2))

	// Set the scaling.
	SetScale/P x,DimOffset(L,0),DimDelta(L,0),detH
	SetScale/P y,DimOffset(L,1),DimDelta(L,1),detH
	SetScale/P z,DimOffset(L,2),DimDelta(L,2),detH

	// Gamma normalize and account for pixel spacing.
	Multithread detH *= (DimOffset(L,2)*DimDelta(L,2)^r)^(2*gammaNorm) / (DimDelta(L,0)*DimDelta(L,1))^2
	
	// Fix the boundary issues again.
	FixBoundaries(detH)
	
	// Clean up
	KillWaves/Z LxxKernel, LyyKernel, LxyKernel, Lxx, Lyy
	
	Return 0
End

// Uses Otsu's method to automatically define a threshold blob strength.
//		detH : The determinant of Hessian blob detector.
//		L : The scale-space representation.
//		doHoles : If 0, only maximal blob reponses are considered. If 1, will consider positive and negative extrema.
//					 * Note this parameter doesn't matter for the determinant of the Hessian since both positive and negative
//					   blobs produce maxima of the determinant of the Hessian.
Function OtsuThreshold(detH,LG,particleType,maxCurvatureRatio)
	Wave detH,LG
	Variable particleType, maxCurvatureRatio
	
	// First identify the maxes
	Wave Maxes = Maxes(detH,LG,particleType,maxCurvatureRatio)
	Duplicate/O Maxes, SS_OTSU_COPY
	Wave Workhorse = SS_OTSU_COPY
	
	// Create a histogram using of the maxes
	Histogram/B=5 /Dest=Hist Maxes
	
	// Search for the best threshold
	Variable i,lim=DimSize(Hist,0)
	Variable minICV=inf, bestThresh = -inf, ICV, xThresh // ICV is intra-class variance
	For(i=0;i<lim;i+=1)
	
		xThresh = DimOffset(Hist,0)+i*DimDelta(Hist,0)

		Multithread Workhorse = SelectNumber(Maxes < xThresh, NaN, Maxes)
		ICV = Sum(Hist,-inf,xThresh)*Variance(Workhorse)
		
		Multithread Workhorse = SelectNumber(Maxes >= xThresh, NaN, Maxes)
		ICV += Sum(Hist,xThresh,inf)*Variance(Workhorse)
	
		If( ICV < minICV )
			bestThresh = xThresh
			minICV = ICV
		EndIf
	
	EndFor
	
	KillWaves/Z Maxes, Hist, Workhorse
	Return bestThresh
End

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

						// PARTICLE MEASUREMENTS //
						
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

// Measure the average pixel value of the particle on the boundary of the particle.
//		im : The image containing the particle.
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
Function M_AvgBoundary(im,mask)
	Wave im, mask
	
	Variable i,j,limI=DimSize(im,0),limJ=DimSize(im,1)
	Variable bg=0,cnt=0
	For(i=1;i<limI-1;i+=1)
	For(j=1;j<limJ-1;j+=1)
		If( mask[i][j]==0 && (mask[i+1][j]==1 || mask[i-1][j]==1 || mask[i][j+1]==1 || mask[i][j-1]==1) )
			bg+=im[i][j]
			cnt+=1
		ElseIf( mask[i][j]==1 && (mask[i+1][j]==0 || mask[i-1][j]==0 || mask[i][j+1]==0 || mask[i][j-1]==0) )
			bg+=im[i][j]
			cnt+=1
		EndIf
	EndFor
	EndFor
	
	If( cnt>0 )
		Return bg/cnt
	Else
		Return 0
	EndIf
End

// Measure the minimum pixel value of the particle on the boundary of the particle.
//		im : The image containing the particle.
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
Function M_MinBoundary(im,mask)
	Wave im, mask
		
	Variable i,j,limI=DimSize(im,0),limJ=DimSize(im,1)
	Variable bg=inf
	For(i=1;i<limI-1;i+=1)
	For(j=1;j<limJ-1;j+=1)
		If( mask[i][j]==1 && im[i][j]<bg )
			bg = im[i][j]
		EndIf
	EndFor
	EndFor
		
	Return bg
End

// Measures the maximum height of the particle above the background level.
//		im : The image containing the particle.
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
//		bg : The background level of the particle.
//		negParticle : An optional parameter that indicates the feature is a hole and has negative height.
Function M_Height(im,mask,bg,[negParticle])
	Wave im,mask
	Variable bg,negParticle
	
	Multithread mask = mask ? im : NaN
	Variable height
	If(ParamIsDefault(negParticle))
	 	height = WaveMax(mask) - bg
	Else
		height = bg - WaveMin(mask)
	EndIf
	
	Multithread mask = NumType(mask)==0 ? 1 : 0
	
	Return height
End

// Computes the volume of the particle.
//		im : The image containing the particle.
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
//		bg : The background level for the particle.
Function M_Volume(im,mask,bg)
	Wave im,mask
	Variable bg
	
	// Find the volume and area
	Variable V=0,cnt=0
	Variable i,j,LimX=DimSize(im,0),LimY=DimSize(im,1)
	For(i=0;i<LimX;i+=1)
	For(j=0;j<LimY;j+=1)
		If(mask[i][j])
			V += im[i][j]
			cnt += 1
		EndIf
	EndFor
	EndFor	
	
	V -= cnt*bg
	V *= DimDelta(im,0) * DimDelta(im,1)
	
	Return V	
End

// Computes the center of mass of the particle, returning the x center of mass and y
// center of mass in a single complex variable COM. The X center of mass is stored in the real part
// and the Y center of mass in the imaginary part. Explicitly, the X part is given by Real(COM) and
// the imaginary part by Imag(COM).
//		im : The image containing the particle.
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
//		bg : The background level for the particle. 
Function/C M_CenterOfMass(im,mask,bg)
	Wave im,mask
	Variable bg
	
	Variable xsum=0,ysum=0,count=0
	Variable i,j,limI=DimSize(im,0),limJ=DimSize(im,1)
	Variable x0=DimOffset(im,0),dx=DimDelta(im,0),y0=DimOffset(im,1),dy=DimDelta(im,1)
	For(i=0;i<limI;i+=1)
	For(j=0;j<limJ;j+=1)
		If(mask[i][j])
			xsum += (x0+i*dx)*(im[i][j]-bg)
			ysum += (y0+j*dy)*(im[i][j]-bg)
			count += (im[i][j]-bg)
		EndIf
	EndFor
	EndFor
	
	Return Cmplx(xsum/count,ysum/count)
End

// Computes the area of the particle using the method employed by Gwyddion:
// http://gwyddion.net/documentation/user-guide-en/grain-analysis.html
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
Function M_Area(mask)
	Wave mask
	
	Variable a=0, pixels
	Variable i,j,limI=DimSize(mask,0),limJ=DimSize(mask,1)
	For(i=0;i<limI-1;i+=1)
	For(j=0;j<limJ-1;j+=1)
	
		pixels = mask[i][j]+mask[i+1][j]+mask[i][j+1]+mask[i+1][j+1]
		
		If(pixels==1)
			a += 0.125  // 1/8
		ElseIf(pixels==2)
			
			If( mask[i][j]==mask[i+1][j] || mask[i][j]==mask[i][j+1] )
				a += 0.5
			Else
				a += 0.75
			EndIf
		
		ElseIf(pixels==3)
			a += 0.875 // 7/8
		ElseIf(pixels==4)	
			a += 1
		EndIf
		
	EndFor
	EndFor
	
	Return a*DimDelta(mask,0)^2
End

// Computes the perimeter of the particle using the method employed by Gwyddion:
// http://gwyddion.net/documentation/user-guide-en/grain-analysis.html
//		mask : A mask image of the same size identifying which pixels belong to the particle.
//				 In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
Function M_Perimeter(mask)
	Wave mask
	
	Variable l=0, pixels
	Variable i,j,limI=DimSize(mask,0),limJ=DimSize(mask,1)
	For(i=0;i<limI-1;i+=1)
	For(j=0;j<limJ-1;j+=1)
	
		pixels = mask[i][j]+mask[i+1][j]+mask[i][j+1]+mask[i+1][j+1]
		
		If(pixels==1 || pixels==3)
			l += sqrt(2)/2
		ElseIf(pixels==2)
			
			If( mask[i][j]==mask[i+1][j] || mask[i][j]==mask[i][j+1] )
				l += 1
			Else
				l += sqrt(2)
			EndIf
			
		EndIf
		
	EndFor
	EndFor
	
	Return l*DimDelta(mask,0)
End

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

						// PREPROCESSING FUNCTIONS //
						
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

// Allows you to preprocess multiple images in a data folder successively. Make sure to highlight the
// folder containing the images in the data browser before executing. 
Function BatchPreprocess()
	
	String ImagesDF=GetBrowserSelection(0)
	String CurrentDF=GetDataFolder(1)
	If( !DataFolderExists(ImagesDF) || CountObjects(ImagesDF,1)<1 )
		DoAlert 0,"Select the folder with your images in it in the data browser, then try again."
		Return -1
	EndIf
	SetDataFolder $ImagesDF
	
	// Declare algorithm parameters.
	Variable flattenOrder = 2
	Variable streakRemovalSDevs = 3
	
	// Retrieve parameters from user
	Prompt flattenOrder,"Polynomial order for flattening?" 
	Prompt streakRemovalSDevs, "Std. Deviations for streak removal?"
	DoPrompt "Preprocessing Parameters", streakRemovalSDevs, flattenOrder
	If( V_Flag == 1 )
		Return -1
	EndIf
	
	// Preprocess the images
	Variable NumImages=CountObjects(ImagesDF,1),i
	For(i=0;i<NumImages;i+=1)
		Wave Im=WaveRefIndexedDFR($ImagesDF,i)
		If(streakRemovalSDevs>0)
			RemoveStreaks(Im,sigma=streakRemovalSDevs)
		EndIf
		If(flattenOrder>0)
			Flatten(Im,flattenOrder)
		EndIf
	EndFor
	
	SetDataFolder $CurrentDF
	Return 0
End

// Flattens ever horizontal line scan of the image by subtracting off a least-squares
//	fitted polynomial of a given order.
//		im : The image to be flattened.
//		order : The order of the polynomial to be subtracted off.
//		mask : An optional mask identifying pixels to fit the polynomial to.
//				 In the mask, 1 corresponds to pixels which will be used for fitting, 0 for pixels to be ignored.
//		noThresh : An optional parameter, if given any value will not prompt the user to set the threshold level.
Function Flatten(im,order,[mask,noThresh])
	Wave im, mask
	Variable order,noThresh
	
	// Want to interactively determine a threshold?
	If( ParamIsDefault(noThresh) )
	
		// Display image and a duplicate for masking
		NewImage/N=IMAGE im 
		AppendImage/T im
		Duplicate/O im, FLATTEN_DUP
		Wave Dup = FLATTEN_DUP
		AppendImage/T Dup
		ModifyImage FLATTEN_DUP ctab={Mean(Im),Mean(Im),Grays,0},minRGB=NaN,maxRGB=(16385,28398,65535,26214)
		
		// Build a Panel with controls
		NewPanel/EXT=0 /HOST=IMAGE /N=SubControl /W=(0,0,100,550) as "Continue Button"
		Button btn pos={0,0}, size={100,50}, title="Accept", win=IMAGE#SubControl, proc=FlattenButton
		Variable/G FLATTEN_THRESH = Mean(Im)
		Slider ThreshSlide limits={WaveMin(Im),WaveMax(Im),(WaveMax(Im)-WaveMin(Im))/300},pos={0,50},size={100,500},variable=FLATTEN_THRESH, proc=FlattenSlider
		
		// Let the user pick the appropriate threshold
		PauseForUser IMAGE
		
		// Make the mask wave
		Wave Mask = Dup
		Mask = Im <= FLATTEN_THRESH
		KillVariables/Z FLATTEN_THRESH
		Print "Flatten Height Threshold: "+Num2Str(FLATTEN_THRESH)
	
	EndIf
	
	// Make a 1D wave for fitting and masking
	Make/N=(DimSize(im,0)) /O FLATTEN_SCANLINE, FLATTEN_MASK
	Wave Scanline = FLATTEN_SCANLINE
	SetScale/P x,DimOffset(im,0),DimDelta(im,0),Scanline
	Wave Mask1D = FLATTEN_MASK
	Mask1D = 1
	
	// Make the coefficient wave
	Make/N=(max(2,order+1)) /O FLATTEN_COEFS=0
	Wave Coefs = FLATTEN_COEFS
	
	// Fit to each scan line
	Variable i,lines=DimSize(im,1)
	For(i=0;i<lines;i+=1)
		
		Multithread Scanline = im[p][i]
		If( !ParamIsDefault(Mask) || ParamIsDefault(noThresh) )
			Mask1D = Mask[p][i]
		EndIf
		
		// Do a fit to the scan line
		If( Order==1 )
			CurveFit/W=2 /Q line, kwCWave=Coefs, Scanline /M=Mask1D
			im[][i] -= Coefs[0] + x*Coefs[1]
		ElseIf( Order==0 )
			CurveFit/W=2 /Q /H="01" line, kwCWave=Coefs, ScanLine /M=Mask1D
			im[][i] -= Coefs[0]
		ElseIf( Order>1 )
			CurveFit/W=2 /Q Poly Order+1, kwCWave=Coefs, Scanline /M=Mask1D
			im[][i] -= Poly(Coefs,x)
		EndIf
		
	EndFor
	
	If( ParamIsDefault(noThresh) )
		KillWaves/Z Mask
	EndIf
	
	KillWaves/Z Scanline,Coefs, Mask1D, :W_ParamConfidenceInterval, :W_sigma
	Return 0
End

// Removes streak artifacts from the image.
//		im : The image from which streaks will be removed.
//		sigma : The number of standard deviations away from mean streak level to smooth a streak.
Function RemoveStreaks(image,[sigma])
	Wave image
	Variable sigma
		If( ParamIsDefault(sigma))
			sigma = 3
			Prompt sigma,"Minimum sigmas from average to smooth a streak?"
			DoPrompt "Parameters",sigma
				If(V_flag==1)
					Return 0
				EndIf
		EndIf
	
	// Produce the dY map
	Wave dyMap=dyMap(image)
	dyMap = abs(dyMap)
	WaveStats/Q dyMap
		Variable MaxDY=V_avg+V_SDev*Sigma
		Variable AvgDY=V_avg
		
	Variable i,j,limI=DimSize(image,0),limJ=DimSize(image,1)-1,i0
	For(i=0;i<limI;i+=1)
	For(j=1;j<limJ;j+=1)
	
		If( dyMap[i][j]>MaxDY )
			i0=i
			
			Do // Go left until the left side of the streak is gone
				image[i][j]=(image[i][j+1]+image[i][j-1])/2
				dyMap[i][j]=0
				
				If( i>0)
					i-=1
				Else
					Break
				EndIf
			While(dyMap[i][j]>AvgDY)
			i=i0
			
			Do // Go right from the original point doing the same thing
				image[i][j]=(image[i][j+1]+image[i][j-1])/2
				dyMap[i][j]=0
				
				If( i<limI-1)
					i+=1
				Else
					Break
				EndIf
			While(dyMap[i][j]>AvgDY)
			i=i0
			
		EndIf
	EndFor
	EndFor
	
	KillWaves/Z dyMap
	CleanWaveStats()
	
	Return 0
End

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

						// UTILITIES //
						
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

// Fixes a boundary issue in the blob detectors. Arises from trying to measure derivatives on the boundary.
//		detH : The determinant of Hessian blob detector, but also works for the Laplacian of Gaussian.
Function FixBoundaries(detH)
	Wave detH
	
	Variable i,limP=DimSize(detH,0)-1,limQ=DimSize(detH,1)-1
	
	// Do the sides first. Corners need extra care.
	// Make the edges fade off so that maxima can still be detected.
	For(i=2;i<=limP-2;i+=1)
		Multithread detH[i][0][] = detH[i][2][r]/3
		Multithread detH[i][1][] = detH[i][2][r]*2/3
	EndFor
	For(i=2;i<=limP-2;i+=1)
		Multithread detH[i][limQ][]   = detH[i][limQ-2][r]/3
		Multithread detH[i][limQ-1][] = detH[i][limQ-2][r]*2/3
	EndFor
	For(i=2;i<=limQ-2;i+=1)
		Multithread detH[0][i][] = detH[2][i][r]/3
		Multithread detH[1][i][] = detH[2][i][r]*2/3
	EndFor
	For(i=2;i<=limQ-2;i+=1)
		Multithread detH[limQ][i][]   = detH[limQ-2][i][r]/3
		Multithread detH[limQ-1][i][] = detH[limQ-2][i][r]*2/3
	EndFor
	
	// Top Left Corner
	Multithread detH[1][1][] = (detH[1][2][r]+detH[2][1][r])/2
	Multithread detH[1][0][] = (detH[1][1][r]+detH[2][0][r])/2
	Multithread detH[0][1][] = (detH[1][1][r]+detH[0][2][r])/2
	Multithread detH[0][0][] = (detH[0][1][r]+detH[1][0][r])/2
	
	// Bottom Right Corner
	Multithread detH[limP-1][limQ-1][] = (detH[limP-1][limQ-2][r]+detH[limP-2][limQ-1][r])/2
	Multithread detH[limP-1][limQ][] = (detH[limP-1][limQ-1][r]+detH[limP-2][limQ][r])/2
	Multithread detH[limP][limQ-1][] = (detH[limP-1][limQ-1][r]+detH[limP][limQ-2][r])/2
	Multithread detH[limP][limQ][] = (detH[limP-1][limQ][r]+detH[limP][limQ-1][r])/2
	
	// Top Right Corner
	Multithread detH[limP-1][1][] = (detH[limP-1][2][r]+detH[limP-2][1][r])/2
	Multithread detH[limP-1][0][] = (detH[limP-1][1][r]+detH[limP-2][0][r])/2
	Multithread detH[limP][1][] = (detH[limP-1][1][r]+detH[limP][2][r])/2
	Multithread detH[limP][0][] = (detH[limP-1][0][r]+detH[limP][1][r])/2
	
	// Bottom Left Corner
	Multithread detH[1][limQ-1][] = (detH[1][limQ-2][r]+detH[2][limQ-1][r])/2
	Multithread detH[1][limQ][] = (detH[1][limQ-1][r]+detH[2][limQ][r])/2
	Multithread detH[0][limQ-1][] = (detH[1][limQ-1][r]+detH[0][limQ-2][r])/2
	Multithread detH[0][limQ][] = (detH[1][limQ][r]+detH[0][limQ-1][r])/2
	
	Return 0
End

// Returns a wave with the values of the local maxes of the determinant of Hessian.
Function/Wave Maxes(detH,LG,particleType,maxCurvatureRatio,[map,scaleMap])
	Wave detH,LG,map,scaleMap
	Variable particleType,maxCurvatureRatio
	
	String Name = NameOfWave(detH)+"_MaxValues"
	Make/N=(NumPnts(detH)/26) /O $Name
	Wave Maxes = $Name
	
	Variable i,j,k
	Variable limI=DimSize(detH,0)-1,limJ=DimSize(detH,1)-1,limK=DimSize(detH,2)-1
	Variable strictlyGreater, greaterOrEqual, cnt=0
	
	// Start with smallest blobs then go to larger blobs
	For(k=1;k<limK-1;k+=1)
	
	For(i=1;i<limI;i+=1)
	For(j=1;j<limJ;j+=1)
		
		// Is it too edgy?
		If( LG[i][j][k]^2/detH[i][j][k] >= (maxCurvatureRatio+1)^2/maxCurvatureRatio )
			Continue
		EndIf
		
		// Is it the right type of particle?
		If( (particleType==-1 && LG[i][j][k]<0) || (particleType==1 && LG[i][j][k]>0)  )
			Continue
		EndIf
	
		// There are 26 neighbors in three dimensions.. Have to check if it is a local maximum
		strictlyGreater = max(detH[i-1][j-1][k-1],max(detH[i-1][j-1][k],max(detH[i-1][j][k-1],detH[i][j-1][k-1])))
		strictlyGreater = max(strictlyGreater,max(detH[i][j][k-1],max(detH[i][j-1][k],detH[i-1][j][k])))
		
		If( !(detH[i][j][k]>strictlyGreater) )
			Continue
		EndIf 
		
		greaterOrEqual = detH[i-1][j-1][k+1]
		greaterOrEqual = max(greaterOrEqual , detH[i-1][j][k+1]) 
		greaterOrEqual = max(greaterOrEqual , max(detH[i-1][j+1][k-1] , max(detH[i-1][j+1][k] , detH[i-1][j+1][k+1])))
		
		greaterOrEqual = max(greaterOrEqual , detH[i][j-1][k+1])
		greaterOrEqual = max(greaterOrEqual , detH[i][j][k+1])
		greaterOrEqual = max(greaterOrEqual ,max( detH[i][j+1][k-1] ,max( detH[i][j+1][k] , detH[i][j+1][k+1])))
		
		greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j-1][k-1] , max(detH[i+1][j-1][k] , detH[i+1][j-1][k+1])))
		greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j][k-1] , max(detH[i+1][j][k] , detH[i+1][j][k+1])))
		greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j+1][k-1] , max(detH[i+1][j+1][k] , detH[i+1][j+1][k+1])))
		
		If( !(detH[i][j][k]>=greaterOrEqual) )
			Continue
		EndIf
		
		Maxes[cnt] = detH[i][j][k]
		cnt+=1
		
		If( !ParamIsDefault(map) )
			map[i][j] = max(map[i][j],detH[i][j][k])
		EndIf
		
		If( !ParamIsDefault(scaleMap) )
			scaleMap[i][j] = DimOffset(detH,2)*(DimDelta(detH,2)^k)
		EndIf
		
	EndFor
	EndFor
	EndFor
	
	DeletePoints cnt,NumPnts(detH),Maxes
	
	Return Maxes
End

// Lets the user interactively choose a blob strength for the determinant of Hessian.
//		im : The image under analysis.
//		detH : The determinant of Hessian blob detector.
//		LG : The Laplacian of Gaussian blob detector.
//		particleType : 1 to consider positive Hessian blobs, 0 to consider negative Hessian blobs.
//		maxCurvatureratio : Maximum ratio of the principal curvatures of a blob
Function InteractiveThreshold(im,detH,LG,particleType,maxCurvatureRatio)
	Wave im,detH,LG
	Variable particleType,maxCurvatureRatio
	
	// First identify the maxes
	Duplicate/O im, SS_MAXMAP
	Wave Map = SS_MAXMAP
	Multithread Map = -1
	Duplicate/O Map, SS_MAXSCALEMAP
	Wave ScaleMap = SS_MAXSCALEMAP
	Wave Maxes = Maxes(detH,LG,particleType,maxCurvatureRatio,map=Map,scaleMap=ScaleMap)
	Maxes = Sqrt(Maxes) // Put it into image units

	// Display image and a duplicate for masking
	NewImage/N=IMAGE im
		
	// Build a Panel with controls
	NewPanel/EXT=0 /HOST=IMAGE /N=SubControl /W=(0,0,200,550) as "Continue Button"
	Button btn pos={0,0}, size={100,50}, title="Accept", win=IMAGE#SubControl, proc=InteractiveContinue
	Button btnQuit pos={100,0}, size={100,50}, title="Quit", win=IMAGE#SubControl, proc=InteractiveQuit
	Variable/G SS_THRESH = WaveMax(Maxes)/2
	Slider ThreshSlide limits={0,WaveMax(Maxes)*1.1,WaveMax(Maxes)*1.1/200},pos={0,80},size={100,470},ticks=10,variable=SS_THRESH, proc=InteractiveSlider
	SetVariable ThreshSetVar limits={0,WaveMax(Maxes)*1.1,WaveMax(Maxes)*1.1/200},pos={10,50},size={170,100},variable=SS_THRESH, title="Blob Strength", proc=InteractiveSetVar
		
	// Let the user pick the appropriate threshold
	PauseForUser IMAGE
	
	Variable returnVal = SS_THRESH
	KillVariables/Z SS_THRESH
	KillWaves/Z Map

	Return returnVal
End

Function InteractiveContinue(B_Struct) : ButtonControl
	STRUCT WMButtonAction &B_Struct
	
	If( B_Struct.eventCode==2 )	
		KillWindow/Z IMAGE
	EndIf
	
	Return 0
End

Function InteractiveQuit(B_Struct) : ButtonControl
	STRUCT WMButtonAction &B_Struct
	
	If( B_Struct.eventCode==2 )	
		KillWindow/Z IMAGE
		String df = GetDataFolder(1)
		KillDataFolder/Z $df
		Abort
	EndIf
	
	Return 0
End

Function InteractiveSlider(S_Struct) : SliderControl
	STRUCT WMSliderAction &S_Struct
	
	If(S_Struct.eventcode==9)
		
		SetDrawLayer/K /W=IMAGE overlay
		Wave Map = :SS_MAXMAP
		Wave ScaleMap = :SS_MAXSCALEMAP
		Variable i,j,limI=DimSize(Map,0),limJ=DimSize(Map,1),rad,xc,yc
		For(i=0;i<limI;i+=1)
		For(j=0;j<limJ;j+=1)
		
			If(Map[i][j]>S_STruct.curval^2)
			
				xc = DimOffset(map,0)+i*DimDelta(map,0)
				yc = DimOffset(map,1)+j*DimDelta(map,1)
				rad = sqrt(2*ScaleMap[i][j])
				
				SetDrawEnv/W=IMAGE xcoord=top, ycoord=left, linethick=2, linefgc=(65535,0,0), fillpat=0
				DrawOval/W=IMAGE xc-rad,yc+rad,xc+rad,yc-rad
			
			EndIf
		
		EndFor
		EndFor
	
	EndIf	
	
	Return 0
End

Function InteractiveSetVar(SV_Struct) : SetVariableControl
	STRUCT WMSetVariableAction &SV_Struct
	
	If(SV_Struct.eventcode==1 || SV_Struct.eventcode==2 )
		
		SetDrawLayer/K /W=IMAGE overlay
		Wave Map = :SS_MAXMAP
		Wave ScaleMap = :SS_MAXSCALEMAP
		Variable i,j,limI=DimSize(Map,0),limJ=DimSize(Map,1),rad,xc,yc
		For(i=0;i<limI;i+=1)
		For(j=0;j<limJ;j+=1)
		
			If(Map[i][j]>SV_Struct.dval^2)
			
				xc = DimOffset(map,0)+i*DimDelta(map,0)
				yc = DimOffset(map,1)+j*DimDelta(map,1)
				rad = sqrt(2*ScaleMap[i][j])
				
				SetDrawEnv/W=IMAGE xcoord=top, ycoord=left, linethick=2, linefgc=(65535,0,0), fillpat=0
				DrawOval/W=IMAGE xc-rad,yc+rad,xc+rad,yc-rad
			
			EndIf
		
		EndFor
		EndFor
	
	EndIf	
	
	Return 0
End

// It's a speed demon, about 10x faster than iterative seedfill and more flexible as well.
// The dest wave should be -1 at background locations, and some positive value where a particle is identified.
// If two fills collide with each other, the one with the higher fill value will continue while the other is deleted.
Function/C ScanlineFill8_LG(image,dest,LG,seedP,seedQ,Thresh,[SeedStack,BoundingBox,fillVal,fillDown,perimeter,x0,xf,y0,yf,layer,dest2,fillVal2,destLayer]) // Destwave and perimeter should be the same size as image
	Wave image,dest,LG,perimeter,SeedStack,BoundingBox,dest2
	Variable seedP,seedQ,Thresh,fillVal,fillDown,x0,xf,y0,yf,layer,fillVal2,destLayer
	Variable fill,i,i0,j,count=0,isBP=0	
	
	// Get the parameters straight
   If( ParamIsDefault(fillVal) )
   		fill=1
  	Else
   		fill=fillVal
 	EndIf
  	If( !ParamIsDefault(fillDown) )
		image *= -1
		thresh *= -1
  	EndIf
  	If(ParamIsDefault(x0) || ParamIsDefault(xf))
  		x0=0
  		xf=DimSize(image,0)-1
  	Else
  		x0 = Max(0,Min(DimSize(image,0)-1,Round(x0)))
  		xf = Max(x0,Min(DimSize(image,0)-1,Round(xf)))
  	EndIf
  	If(ParamIsDefault(y0) || ParamIsDefault(yf))
  		y0=0
  		yf=DimSize(image,1)-1
  	Else
  		y0 = Max(0,Min(DimSize(image,1)-1,Round(y0)))
  		yf = Max(y0,Min(DimSize(image,1)-1,Round(yf)))
  	EndIf
  	If(ParamIsDefault(layer))
  		layer=0
  	EndIf
  	If(ParamIsDefault(destLayer))
  		destLayer = 0
  	EndIf
  	
  	// Optionally fill a second destination wave with a different fill value
  	Variable doDest2=0
  	If(!ParamIsDefault(dest2))
  		doDest2=1
  	EndIf
  	
  	// Make sure the seeds are in bounds, and that the seed is valid 
 	If( seedP<x0 || seedQ<y0 || seedP>xf || seedQ>yf )
   	Return Cmplx(0,-3)
   ElseIf(image[seedP][seedQ][layer]<=thresh)
      	Return Cmplx(0,-1)
 	EndIf
 	
 	// Get the sign of the LG
 	Variable sgn = sign(LG[seedP][seedQ][layer])
   
   // Make a pseudo stack
   Variable killStack = 0
   If( ParamIsDefault(SeedStack) )
   		String name = UniqueName("SeedStack",1,0)
   		Make/N=((xf-x0)*(yf-y0),4) $name // Col 0:i0, Col 1:i, Col 2:j, Col 3:state
   		Wave SeedStack = $name
   		killStack = 1
   EndIf
 		SeedStack[0][0]=seedP
 		SeedStack[0][1]=seedP
 		SeedStack[0][2]=seedQ
 		SeedStack[0][3]=0
 	
 	// Keep track of a bounding box for the filled region
 	Variable maxP=seedP,minP=seedP,maxQ=seedQ,minQ=seedQ
 	
 	Variable seedIndex, newSeedIndex, state, goFish
 		seedIndex = 0
 		newSeedIndex = 1
 		state = 0
 		goFish = 1
 			
 	Do
 	
		If(goFish)
			i0 = SeedStack[seedIndex][0]
 			i  = SeedStack[seedIndex][1]
 			j  = SeedStack[seedIndex][2]
 			state = SeedStack[seedIndex][3]
 			seedIndex += 1
 		EndIf
 		
 		goFish=1
 		
 		Switch(state)
 			Case 0: // Scan Right and Left
 			
 				// Keep track of bounding box
 				maxP = max(maxP,i)
 				minP = min(minP,i)
 				maxQ = max(maxQ,j)
 				minQ = min(minQ,j)
 			
 				i=i0
 				Do
 					If(i<=xf && image[i][j][layer] >= thresh && sign(LG[i][j][layer])==sgn)
 						dest[i][j][destLayer]=fill
 						If(doDest2)
 							dest2[i][j][destLayer]=fillVal2
 						EndIf
 						count+=1
 						
 						maxP = max(maxP,i)
 						i+=1
 						
 						//DoUpdate
 						//Sleep/T 8
 					Else	
 						Break
 					EndIf
 				While(1)
 				
 				i=i0-1
 				Do
 					If(i>=x0 && image[i][j][layer] >= thresh && sign(LG[i][j][layer])==sgn)
 						dest[i][j][destLayer]=fill
 						If(doDest2)
 							dest2[i][j][destLayer]=fillVal2
 						EndIf
 						count+=1
 						
 						minP = min(minP,i)
 						i-=1
 						
 						//DoUpdate
 						//Sleep/T 8
 					Else	
 						Break
 					EndIf
 				While(1)
 				
 				i=i0
 				
 			Case 1: // Search Up Right
 				If(j!=yf)
 					Do
 						If(dest[i][j+1][destLayer]!=fill && image[i][j+1][layer] >= thresh && sign(LG[i][j+1][layer])==sgn)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 1
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j+=1
 							Break
 						EndIf
 						
 						i+=1
 					While(i<=xf && dest[i-1][j][destLayer]==fill)
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 2: // Search Up Left
 				If(j!=yf)
 					Do
 						If(dest[i][j+1][destLayer]!=fill && image[i][j+1][layer] >= thresh && sign(LG[i][j+1][layer])==sgn)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 2
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j+=1
 							Break
 						EndIf
 						
 						i-=1
 					While(i>=x0 && dest[i+1][j][destLayer]==fill)
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 3: // Search Down Right
 				If(j!=y0)
 					Do
 						If(dest[i][j-1][destLayer]!=fill && image[i][j-1][layer] >= thresh && sign(LG[i][j-1][layer])==sgn)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 3
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j-=1
 							Break
 						EndIf
 						
 						i+=1
 					While(i<=xf && dest[i-1][j][destLayer]==fill)				
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 4: // Search Down Left
 				If(j!=y0)
 					Do
 						If(dest[i][j-1][destLayer]!=fill && image[i][j-1][layer] >= thresh && sign(LG[i][j-1][layer])==sgn)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 4
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j-=1
 							Break
 						EndIf
 						
 						i-=1
 					While(i>=x0 && dest[i+1][j][destLayer]==fill)			
 				EndIf	
 				
 				i=i0
 			
 		EndSwitch
 	While(seedIndex != newSeedIndex)
 	
 	If( !ParamIsDefault(fillDown) )
		image *= -1
  	EndIf
  	
  	// Check if boundary particle
	For(i=x0;i<=xf;i+=1)
  		If(dest[i][y0][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	If(!isBP)
	For(i=x0;i<=xf;i+=1)
  		If(dest[i][yf][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	If(!isBP)
	For(j=y0;j<=yf;j+=1)
  		If(dest[x0][j][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	If(!isBP)
	For(j=y0;j<=yf;j+=1)
  		If(dest[xf][j][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	
	If( killStack )
		KillWaves/Z SeedStack
	EndIf
	
	If( !ParamIsDefault(BoundingBox) )
		BoundingBox[0] = minP
		BoundingBox[1] = maxP
		BoundingBox[2] = minQ
		BoundingBox[3] = maxQ
	EndIf
	
	KillWaves/Z SCANFILLDUP
  	
  	Return Cmplx(Count,isBP) // Dont forget is BP
End

Function/C ScanlineFillEqual(image,dest,seedP,seedQ,[SeedStack,BoundingBox,fillVal,perimeter,x0,xf,y0,yf,layer,dest2,fillVal2,destLayer]) // Destwave and perimeter should be the same size as image
	Wave image,dest,perimeter,SeedStack,BoundingBox,dest2
	Variable seedP,seedQ,fillVal,x0,xf,y0,yf,layer,fillVal2,destLayer
	Variable fill,i,i0,j,count=0,isBP=0	
	
	// Get the parameters straight
   If( ParamIsDefault(fillVal) )
   		fill=1
  	Else
   		fill=fillVal
 	EndIf
  	If(ParamIsDefault(x0) || ParamIsDefault(xf))
  		x0=0
  		xf=DimSize(image,0)-1
  	Else
  		x0 = Max(0,Min(DimSize(image,0)-1,Round(x0)))
  		xf = Max(x0,Min(DimSize(image,0)-1,Round(xf)))
  	EndIf
  	If(ParamIsDefault(y0) || ParamIsDefault(yf))
  		y0=0
  		yf=DimSize(image,1)-1
  	Else
  		y0 = Max(0,Min(DimSize(image,1)-1,Round(y0)))
  		yf = Max(y0,Min(DimSize(image,1)-1,Round(yf)))
  	EndIf
  	If(ParamIsDefault(layer))
  		layer=0
  	EndIf
  	If(ParamIsDefault(destLayer))
  		destLayer = 0
  	EndIf
  	
  	// Optionally fill a second destination wave with a different fill value
  	Variable doDest2=0
  	If(!ParamIsDefault(dest2))
  		doDest2=1
  	EndIf
  	
  	// Make sure the seeds are in bounds, and that the seed is valid 
 	If( seedP<x0 || seedQ<y0 || seedP>xf || seedQ>yf )
   	Return Cmplx(0,-3)
 	EndIf
 
 	// The value to seed fill	
 	Variable val = image[seedP][seedQ][layer]
   
   // Make a pseudo stack
   Variable killStack = 0
   If( ParamIsDefault(SeedStack) )
   		String name = UniqueName("SeedStack",1,0)
   		Make/N=((xf-x0)*(yf-y0),4) $name // Col 0:i0, Col 1:i, Col 2:j, Col 3:state
   		Wave SeedStack = $name
   		killStack = 1
   EndIf
 		SeedStack[0][0]=seedP
 		SeedStack[0][1]=seedP
 		SeedStack[0][2]=seedQ
 		SeedStack[0][3]=0
 	
 	// Keep track of a bounding box for the filled region
 	Variable maxP=seedP,minP=seedP,maxQ=seedQ,minQ=seedQ
 	
 	Variable seedIndex, newSeedIndex, state, goFish
 		seedIndex = 0
 		newSeedIndex = 1
 		state = 0
 		goFish = 1
 			
 	Do

		If(goFish)
			i0 = SeedStack[seedIndex][0]
 			i  = SeedStack[seedIndex][1]
 			j  = SeedStack[seedIndex][2]
 			state = SeedStack[seedIndex][3]
 			seedIndex += 1
 		EndIf
 		
 		goFish=1
 		
 		Switch(state)
 			Case 0: // Scan Right and Left
 			
 				// Keep track of bounding box
 				maxP = max(maxP,i)
 				minP = min(minP,i)
 				maxQ = max(maxQ,j)
 				minQ = min(minQ,j)
 			
 				i=i0
 				Do
 					If(i<=xf && image[i][j][layer] ==val)
 						dest[i][j][destLayer]=fill
 						If(doDest2)
 							dest2[i][j][destLayer]=fillVal2
 						EndIf
 						count+=1
 						
 						maxP = max(maxP,i)
 						i+=1
 						
 						//DoUpdate
 						//Sleep/T 8
 					Else	
 						Break
 					EndIf
 				While(1)
 				
 				i=i0-1
 				Do
 					If(i>=x0 && image[i][j][layer] ==val)
 						dest[i][j][destLayer]=fill
 						If(doDest2)
 							dest2[i][j][destLayer]=fillVal2
 						EndIf
 						count+=1
 						
 						minP = min(minP,i)
 						i-=1
 						
 						//DoUpdate
 						//Sleep/T 8
 					Else	
 						Break
 					EndIf
 				While(1)
 				
 				i=i0
 				
 			Case 1: // Search Up Right
 				If(j!=yf)
 					Do
 						If(dest[i][j+1][destLayer]!=fill && image[i][j+1][layer] ==val)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 1
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j+=1
 							Break
 						EndIf
 						
 						i+=1
 					While(i<=xf && dest[i][j][destLayer]==fill)
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 2: // Search Up Left
 				If(j!=yf)
 					Do
 						If(dest[i][j+1][destLayer]!=fill && image[i][j+1][layer] ==val)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 2
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j+=1
 							Break
 						EndIf
 						
 						i-=1
 					While(i>=x0 && dest[i][j][destLayer]==fill)
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 3: // Search Down Right
 				If(j!=y0)
 					Do
 						If(dest[i][j-1][destLayer]!=fill && image[i][j-1][layer] ==val)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 3
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j-=1
 							Break
 						EndIf
 						
 						i+=1
 					While(i<=xf && dest[i][j][destLayer]==fill)				
 				EndIf
 				
 				i=i0
 				
 				If(!goFish)
 					Continue
 				EndIf
 			Case 4: // Search Down Left
 				If(j!=y0)
 					Do
 						If(dest[i][j-1][destLayer]!=fill && image[i][j-1][layer] ==val)
 							SeedStack[newSeedIndex][0] = i0
 							SeedStack[newSeedIndex][1] = i
 							SeedStack[newSeedIndex][2] = j
 							SeedStack[newSeedIndex][3] = 4
 							newSeedIndex += 1
 							state = 0
 							goFish = 0
 							i0=i
 							j-=1
 							Break
 						EndIf
 						
 						i-=1
 					While(i>=x0 && dest[i][j][destLayer]==fill)			
 				EndIf	
 				
 				i=i0
 			
 		EndSwitch
 	While(seedIndex != newSeedIndex)
  	
  	// Check if boundary particle
	For(i=x0;i<=xf;i+=1)
  		If(dest[i][y0][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	If(!isBP)
	For(i=x0;i<=xf;i+=1)
  		If(dest[i][yf][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	If(!isBP)
	For(j=y0;j<=yf;j+=1)
  		If(dest[x0][j][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	If(!isBP)
	For(j=y0;j<=yf;j+=1)
  		If(dest[xf][j][destLayer]==fill)
  			isBP=1
  			Break
  		EndIf	
	EndFor
	EndIf
	
	If( killStack )
		KillWaves/Z SeedStack
	EndIf
	
	If( !ParamIsDefault(BoundingBox) )
		BoundingBox[0] = minP
		BoundingBox[1] = maxP
		BoundingBox[2] = minQ
		BoundingBox[3] = maxQ
	EndIf
	
	KillWaves/Z SCANFILLDUP
  	
  	Return Cmplx(Count,isBP) // Dont forget is BP
End

// The map and info must be fed into the function since Igor doesn't return multiple objects..
// ParticleType is -1 for negative particles only, 1 for positive only, 0 for both
Function FindHessianBlobs(im,detH,LG,minResponse,mapNum,mapLG,mapMax,info,particleType,maxCurvatureRatio) 
	Wave im,detH,LG,mapNum,mapLG,mapMax,info
	Variable minResponse,particleType,maxCurvatureRatio
	
	// Square the minResponse, since the parameter is provided as the square root
	// of the actual minimum detH response so that it is in normal image units
	minResponse = minResponse^2
	
	/// mapNum: Map identifying particle numbers
	Redimension/N=(DimSize(im,0),DimSize(im,1),DimSize(detH,2)) mapNum
	Multithread mapNum=-1
	SetScale/P x,DimOffset(im,0),DimDelta(im,0),mapNum
	SetScale/P y,DimOffset(im,1),DimDelta(im,1),mapNum
	
	// mapLG: Map identifying the value of the LoG at the defined scale, useful for setting the low threshold later
	Redimension/N=(DimSize(im,0),DimSize(im,1),DimSize(detH,2)) mapLG
	Multithread mapLG=0 
	SetScale/P x,DimOffset(im,0),DimDelta(im,0),mapLG
	SetScale/P y,DimOffset(im,1),DimDelta(im,1),mapLG
	
	// mapMax: Map identifying the value of the LoG of the maximum pixel in the particle at the defined scale, useful for setting the high threshold later
	Redimension/N=(DimSize(im,0),DimSize(im,1),DimSize(detH,2)) mapMax
	Multithread mapMax=0
	SetScale/P x,DimOffset(im,0),DimDelta(im,0),mapMax
	SetScale/P y,DimOffset(im,1),DimDelta(im,1),mapMax
		
	// Maintain an info wave with particle boundaries and info
	Redimension/N=(DimSize(im,0)*DimSize(im,1)*DimSize(detH,2)/27,15) info
		// Info[i][0] = P Seed
		// Info[i][1] = Q Seed
		// Info[i][2] = NumPixels
		// Info[i][3] = MaxVal
		// Info[i][4] = pStart
		// Info[i][5] = pStop
		// Info[i][6] = qStart
		// Info[i][7] = qStop
		// Info[i][8] = scale
		// Info[i][9] = layer in L
		// Info[i][10] = 1 for maximal over scales and overlapped space, 0 else
		// Info[i][11] = Number of parent blob, own number if maximal
		// Info[i][12] = Number of blobs contained in support if maximal, else 0
		// Info[i][13] = -1 for edges that aren't well defined, 1 for well defined, 0 for undetermined
		// Info[i][14] = -1 for particle rejected, 0 for undetermined, 1 for particle accepted
	
	// Make a bounding box wave for scanfill
	Make/N=4 /O SSPARTICLESBOX
	Wave Box=SSPARTICLESBOX
	
	Variable i,j,k,skip
	Variable limI=DimSize(detH,0)-1,limJ=DimSize(detH,1)-1,limK=DimSize(detH,2)-1
	Variable strictlyGreater, greaterOrEqual, radius, cnt=0
	Variable/C numPixels
	
	// Start with smallest blobs then go to larger blobs
	For(k=1;k<limK;k+=1)
		
	For(i=1;i<limI-1;i+=1)
	For(j=1;j<limJ-1;j+=1)
	
		// Does it hit the threshold?
		If( detH[i][j][k] < minResponse )
			Continue
		EndIf
		
		// Is it too edgy?
		If( LG[i][j][k]^2/detH[i][j][k] >= (maxCurvatureRatio+1)^2/maxCurvatureRatio )
			Continue
		EndIf
		
		// Is there a particle there already?
		If( mapNum[i][j][k] > -1 && detH[i][j][k] <= info[mapNum[i][j][k]][3])
			Continue
		EndIf
		
		// Is it the right type of particle?
		If( (particleType==-1 && LG[i][j][k]<0) || (particleType==1 && LG[i][j][k]>0)  )
			Continue
		EndIf
	
		// There are 26 neighbors in three dimensions.. Have to check if it is a local maximum
		If(k!=0)
			strictlyGreater = max(detH[i-1][j-1][k-1],max(detH[i-1][j-1][k],max(detH[i-1][j][k-1],detH[i][j-1][k-1])))
			strictlyGreater = max(strictlyGreater,max(detH[i][j][k-1],max(detH[i][j-1][k],detH[i-1][j][k])))
		Else
			strictlyGreater = max(detH[i-1][j-1][k],detH[i][j-1][k],detH[i-1][j][k])
		EndIf
		
		If( !(detH[i][j][k]>strictlyGreater) )
			Continue
		EndIf 
		
		If(k!=0)
			greaterOrEqual = detH[i-1][j-1][k+1]
			greaterOrEqual = max(greaterOrEqual , detH[i-1][j][k+1]) 
			greaterOrEqual = max(greaterOrEqual , max(detH[i-1][j+1][k-1] , max(detH[i-1][j+1][k] , detH[i-1][j+1][k+1])))
		
			greaterOrEqual = max(greaterOrEqual , detH[i][j-1][k+1])
			greaterOrEqual = max(greaterOrEqual , detH[i][j][k+1])
			greaterOrEqual = max(greaterOrEqual ,max( detH[i][j+1][k-1] ,max( detH[i][j+1][k] , detH[i][j+1][k+1])))
		
			greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j-1][k-1] , max(detH[i+1][j-1][k] , detH[i+1][j-1][k+1])))
			greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j][k-1] , max(detH[i+1][j][k] , detH[i+1][j][k+1])))
			greaterOrEqual = max(greaterOrEqual , max(detH[i+1][j+1][k-1] , max(detH[i+1][j+1][k] , detH[i+1][j+1][k+1])))
		Else
			greaterOrEqual = max(detH[i-1][j-1][k+1],detH[i-1][j][k+1],detH[i-1][j+1][k],detH[i-1][j+1][k+1],detH[i][j-1][k+1],detH[i][j][k+1])
			greaterOrEqual = max(greaterOrEqual,detH[i][j+1][k],detH[i][j+1][k+1],detH[i+1][j-1][k],detH[i+1][j-1][k+1])
			greaterOrEqual = max(greaterOrEqual,detH[i+1][j][k],detH[i+1][j][k+1],detH[i+1][j+1][k],detH[i+1][j+1][k+1])
		EndIf
		
		If( !(detH[i][j][k]>=greaterOrEqual) )
			Continue
		EndIf
		
		// It's a local max, is it overlapped and bigger than another one already?
		If( mapNum[i][j][k] > -1 )
			info[mapNum[i][j][k]][0] = i
			info[mapNum[i][j][k]][1] = j
			info[mapNum[i][j][k]][3] = detH[i][j][k]
			Continue
		EndIf
		
		// It's a local max, proceed to fill out the feature.
		//numPixels = EdgeScanFill(Edges,mapMax,i,j,BoundingBox=Box,fillVal=detH[i][j][k],layer=k,dest2=mapNum,fillVal2=cnt,destLayer=k)	
		numPixels = ScanlineFill8_LG(detH,mapMax,LG,i,j,0,BoundingBox=Box,fillVal=detH[i][j][k],layer=k,dest2=mapNum,fillVal2=cnt,destLayer=k)
		If( numPixels==-2 )
			mapMax[i][j][k] = detH[i][j][k]
			mapNum[i][j][k] = cnt
			numPixels = 1
			Box[0]=i
			Box[1]=i
			Box[2]=j
			Box[3]=j
		EndIf
			
		info[cnt][0] = i
		info[cnt][1] = j
		info[cnt][2] = Real(numPixels)
		info[cnt][3] = detH[i][j][k]
		info[cnt][4] = Box[0]
		info[cnt][5] = Box[1]
		info[cnt][6] = Box[2]
		info[cnt][7] = Box[3]
		info[cnt][8] = DimOffset(detH,2)*(DimDelta(detH,2)^k)
		info[cnt][9] = k
		info[cnt][10] = 1
		
		cnt+=1
	
	EndFor
	EndFor
	EndFor
	
	// Remove unused rows in the info wave
	DeletePoints/M=0 cnt,DimSize(im,0)*DimSize(im,1)*DimSize(detH,2),info
	
	// Make the mapLG
	Multithread mapLG = SelectNumber( mapNum != -1 , 0, detH[p][q][r] )
	
	KillWaves/Z Box, SeedStack
	Return 0
End

Function MaximalBlobs(info,mapNum)
	Wave info, mapNum
	
	If(DimSize(info,0)==0)
		Return -1
	EndIf
	
	// Initialize maximality of each particle as undetermined (-1)
	info[][10] = -1
	
	// Make lists for organzing overlapped particles
	Make/N=(DimSize(info,0)) /O MAXPARTICLES_NUMBER, MAXPARTICLES_STRENGTH
	Wave BlobListNumber = MAXPARTICLES_NUMBER
	Wave BlobListStrength = MAXPARTICLES_STRENGTH
	
	// Info Wave Contents Reminder
		// Info[i][0] = P Seed
		// Info[i][1] = Q Seed
		// Info[i][2] = NumPixels
		// Info[i][3] = MaxVal
		// Info[i][4] = pStart
		// Info[i][5] = pStop
		// Info[i][6] = qStart
		// Info[i][7] = qStop
		// Info[i][8] = scale
		// Info[i][9] = layer in L
		// Info[i][10] = 1 for maximal over scales and overlapped space, 0 else
		// Info[i][11] = Number of parent blob, own number if maximal
		// Info[i][12] = Number of blobs contained in support if maximal, else 0
		
	BlobListNumber = p
	BlobListStrength = info[p][3]
	
	// Sort by blob strength
	Sort/R BlobListStrength, BlobListStrength, BlobListNumber
	
	Variable i,lim=DimSize(info,0),index,k,kk,limK=DimSize(mapnum,2)
	Variable ii,jj,blocked
	For(i=0;i<lim;i+=1)
	
		// See if there's room for the i'th strongest particle
		blocked=0
		index=BlobListNumber[i]
		k=info[index][9]
		For(ii=info[index][4] ; ii<=info[index][5] ; ii+=1)
		For(jj=info[index][6] ; jj<=info[index][7] ; jj+=1)
			If( mapNum[ii][jj][k]==index )
			
				For(kk=0;kk<limK;kk+=1)
					If( mapNum[ii][jj][kk]!=-1 && info[mapNum[ii][jj][kk]][10]==1 )
						blocked=1
						Break
					EndIf
				EndFor
				
				If(blocked)
					Break
				EndIf
				
			EndIf
		EndFor
			If(blocked)
				Break
			EndIf
		EndFor
		
		info[index][10] = blocked ? 0 : 1
	
	EndFor
	
	KillWaves/Z BlobListNumber, BlobListStrength
	Return 0
End

Function ExpandBoundary8(mask)
	Wave mask
	
	Multithread mask = (p>0 && p<DimSize(mask,0)-1 && q>0 && q<DimSize(mask,1)-1 && mask[p][q][r]==0 && (mask[p+1][q][r]==1 || mask[p-1][q][r]==1 || mask[p][q+1][r]==1 || mask[p][q-1][r]==1 || mask[p+1][q+1][r]==1 || mask[p-1][q+1][r]==1 || mask[p+1][q-1][r]==1 || mask[p-1][q-1][r]==1) ) ? 2 : mask[p][q][r]
	Multithread mask = mask > 0

	Return 0
End

Function ExpandBoundary4(mask)
	Wave mask
	
	Multithread mask = (p>0 && p<DimSize(mask,0)-1 && q>0 && q<DimSize(mask,1)-1 && mask[p][q][r]==0 && (mask[p+1][q][r]==1 || mask[p-1][q][r]==1 || mask[p][q+1][r]==1 || mask[p][q-1][r]==1) ) ? 2 : mask[p][q][r]
	Multithread mask = mask > 0

	Return 0
End

Threadsafe Function BilinearInterpolate(im,x0,y0,[r0])
	Wave im
	Variable x0,y0,r0
	
	r0 = ParamIsDefault(r0) ? 0 : r0
	
	Variable pMid = (x0-DimOffset(im,0))/DimDelta(im,0)
	Variable p0 = Max(0,Floor(pMid)), p1 = Min(DimSize(im,0)-1,Ceil(pMid))
	Variable qMid = (y0-DimOffset(im,1))/DimDelta(im,1)
	Variable q0 = Max(0,Floor(qMid)), q1 = Min(DimSize(im,1)-1,Ceil(qMid))
	
	Variable pInterp0 = im[p0][q0][r0] + (im[p1][q0][r0]-im[p0][q0][r0])*(pMid-p0)
	Variable pInterp1 = im[p0][q1][r0] + (im[p1][q1][r0]-im[p0][q1][r0])*(pMid-p0)
	
	Return pInterp0 + (pInterp1-pInterp0)*(qMid-q0)
End

Menu "GraphPopup"
	"Scan Line", ImScanLine()
End

Function/Wave ImScanLine([numPoints])
	Variable numPoints
	
	String ImStr = StringFromList(0,ImageNameList("",";"))
	If( !WaveExists(ImageNameToWaveRef("",ImStr)) )
		DoAlert/T="No Image Found" 0,"Could not identify an image in the top graph."
		Abort
	EndIf
	
	Wave im = ImageNameToWaveRef("",ImStr)
	
	// Are there cursors?
	If(StrLen(CsrInfo(A))<2 || StrLen(CsrInfo(B))<2 )
		DoAlert/T="No Cursors Found" 0,"Could not find cursors for the line scan."
		Abort
	EndIf
	
	Variable x1 = XCsr(A)
	Variable y1 = VCsr(A)
	Variable x2 = XCsr(B)
	Variable y2 = VCsr(B)
	
	If(ParamIsDefault(numPoints))
		numPoints=max(200, Round(Sqrt( (abs(x2-x1)/DimDelta(im,0))^2 + (abs(y2-y1)/DimDelta(im,1))^2 ) ))
	EndIf
	
	String Name=NameOfWave(im)+"_ScanLine"
	Wave Scan = $Name
	If( !WaveExists(Scan) )
		Make/N=(numPoints) /O $Name
		Wave Scan = $Name
		Scan = 0
		SetScale/I x,0,Sqrt( (x2-x1)^2 + (y2-y1)^2 ),Scan
	EndIf
	
	Name=NameOfWave(im)+"_OnImage"
	Wave Line=$Name
	If( !WaveExists(Line) )
		Make/N=(2,2) /O $Name
		Wave Line = $Name
			Line[0][0] = x1
			Line[0][1] = y1
			Line[1][0] = x2
			Line[1][1] = y2
			String axis = StringByKey("XAXIS",ImageInfo("",NameOfWave(im),0),":",";")
			If( CmpStr(axis,"top")==0 )
				AppendToGraph/L /T Line[][1] vs Line[][0]
			Else
				AppendToGraph/L /B Line[][1] vs Line[][0]
			EndIf		
	Else
		Line[0][0] = x1
		Line[0][1] = y1
		Line[1][0] = x2
		Line[1][1] = y2
	EndIf
	
	If(WaveDims(im)==2)
		Scan[] = Interp2D(im,x1+(p/(numPoints-1))*(x2-x1),y1+(p/(numPoints-1))*(y2-y1))
	Else
		Variable layer = Str2Num( StringByKey("plane",ImageInfo("",NameOfWave(im),0),"=",";"))
		Scan[] = Interp3D(im,x1+(p/(numPoints-1))*(x2-x1),y1+(p/(numPoints-1))*(y2-y1),layer)
	EndIf
	
	Note/K Scan
	Note Scan,"Win:"+StringFromList(0,WinList("*",";",""))
	Note Scan, "Im:"+NameOfWave(im)
	
	DoWindow ScanLine
	If( !V_flag )
		SetWindow kwTopWin,hook(UpdateHook)=UpdateScanHook,hookevents=4
		Display/N=ScanLine /K=1 Scan as "Scan Line: "+NameOfWave(im)
		ModifyGraph lsize=2
		SetWindow ScanLine, hook(TheHook)=ScanKillHook
	Else
		String win = StringByKey("Win",Note(TraceNameToWaveRef("ScanLine",StringFromList(0,TraceNameList("ScanLine",";",1)))),":","\r")
		If( CmpStr(StringFromList(0,WinList("*",";","")),win) != 0 )
			Print "New Graph"
			DoWindow/K ScanLine
			SetWindow kwTopWin,hook(UpdateHook)=UpdateScanHook,hookevents=4
			Display/N=ScanLine /K=1 Scan as "Scan Line: "+NameOfWave(im)
			ModifyGraph lsize=2
			SetWindow ScanLine, hook(TheHook)=ScanKillHook
		EndIf
	EndIf
	
	Return Scan
End

Function UpdateScanHook(s)
	STRUCT WMWinHookStruct &s
	
	If(s.eventcode==7)
		If(StrLen(CsrInfo(A))>2 && StrLen(CsrInfo(B))>2)
			ImScanLine()
		Else
			KillScan()
		EndIf
	EndIf
End

Function ScanKillHook(s)
	STRUCT WMWinHookStruct &s
	If(s.eventcode==2)
		KillScan()
	EndIf
End

Function KillScan()
Wave Scan = TraceNameToWaveRef("ScanLine",StringFromList(0,TraceNameList("ScanLine",";",1)))
	If(WaveExists(Scan))
		String win = StringByKey("Win",Note(Scan),":","\r")
		DoWindow $win
		If(v_flag==1)
			SetWindow $win, hook(UpdateHook)=$""
		EndIf
		
		String LineName = GetWavesDataFolder(Scan,2)
		LineName = LineName[0,StrLen(LineName)-9]
		LineName += "OnImage"
		Wave Line = $LineName
		If( WaveExists(Line) )
			LineName = NameOfWave(Line)
			DoWindow $win
			If( V_flag )
				RemoveFromGraph/Z /W=$win $LineName
			EndIf
			KillWaves/Z Line
		EndIf
		
		DoWindow/K ScanLine
		KillWaves/Z Scan
	EndIf
End

// 5/22/16
// To use, click on the folder containing the particle folders and run the function.
// The arrow keys can be used to cycle left and right, and hitting space bar brings
// up the delete particle warning. To quickly delete a particle, hit space bar or down arrow then
// enter (works on mac at least). The rest should be self-explanatory, using a value
// of -1 anywhere will result in autoscaling for that option.
// NOTE: The controls panel must be the front window for the keyboard shortcuts to work.
Function ViewParticles()
	
	// See if the data folder is valid
	DFREF ParticlesDF = $GetBrowserSelection(0)
	String ParticlesDFstr = GetBrowserSelection(0)
	If( !DataFolderRefStatus(ParticlesDF)==1 || CountObjectsDFR(ParticlesDF,4)==0 )
		DoAlert 0,"Please select the folder containing the crop folders."
		Return -1
	EndIf
	
	// Set up the viewing panel
	DFREF DF = ParticlesDF:$GetIndexedObjNameDFR(ParticlesDF,4,0)
	String DFstr = ParticlesDFstr+GetIndexedObjNameDFR(ParticlesDF,4,0)
	Wave FirstIm = WaveRefIndexedDFR(DF,0)
	Wave FirstPerim = $(DFstr+":SubPixEdges_"+Num2Str(ParticleNumber(NameOfWave(FirstIm))))
		If( !WaveExists(FirstIm) )
			Return -2
		EndIf
	
	DoWindow/K ParticleView
	Display/K=1 /N=ParticleView /W=(500,200,900,600) as "Particle Viewer"
	AppendImage FirstIm
	AppendImage FirstPerim
	ModifyImage $NameOfWave(FirstPerim) ctab= {0,1,Grays,0}, interpolate=1
	ShowInfo
	NewPanel/HOST=ParticleView /EXT=0 /K=2 /W=(0,0,150,398) /N=ViewControls as "Controls"
		TitleBox ParticleName pos={20,10},size={140,25},fsize=15,fstyle=1,frame=0
		Button NextBtn pos={80,40},size={60,25},title="Next",fsize=13,proc=ViewNextBtn
		Button PrevBtn pos={10,40},size={60,25},title="Prev",fsize=13,proc=ViewPrevBtn
		SetVariable GoTo pos={30,75},size={100,25},title="Go To:",fsize=13,limits={0,inf,0},value=_NUM:0,proc=ViewGoTo
		PopUpMenu ColorTab pos={10,110},size={130,25},bodywidth=130,fsize=13,title="",value="*COLORTABLEPOP*",proc=ViewColorTab
		SetVariable ColorRange pos={10,130},size={127,25},title="Color Range",limits={-inf,inf,1e-10},value=_NUM:-1,proc=ViewColorRange
		Checkbox Interpo pos={10,145},size={100,10},side=1,title="Interpolate:",value=0,proc=ViewInterp
		Checkbox Perim pos={10,160},size={100,10},side=1,title="Perimeter:",value=1,proc=ViewPerim
		SetVariable XRange pos={10,180},size={127,25},title="X-Range:",limits={0,inf,1e-9},value=_NUM:-1,proc=ViewRange
		SetVariable YRange pos={10,195},size={127,25},title="Y-Range:",limits={0,inf,1e-9},value=_NUM:-1,proc=ViewRange
		TitleBox HeightTitle pos={10,220},size={150,25},fsize=15,frame=0,title="Height"
		ValDisplay HeightDisp pos={10,245},size={130,25},fsize=15,frame=3,value=_NUM:0
		TitleBox VolTitle pos={10,285},size={150,25},fsize=15,frame=0,title="Volume"
		ValDisplay VolDisp pos={10,310},size={130,25},fsize=15,frame=3,value=_NUM:0
		Button DeleteBtn pos={10,370},size={130,25},title="DELETE",fsize=14,fstyle=1,fColor=(32000,0,0),proc=ViewDelete

		// Keep track of all the settings in a hidden SetVariable control
		String ViewInfo = "ParticlesDF{"+ParticlesDFstr+"}"
		ViewInfo += "CurrentDF{"+DFstr+":"+"}"
		ViewInfo += "ColorTab{Mud}"
		ViewInfo += "ColorRange{-1}"
		ViewInfo += "Interp{0}"
		ViewInfo += "Perim{1}"
		ViewInfo += "XRange{-1}"
		ViewInfo += "YRange{-1}"
		SetVariable Info pos={0,500},noproc,value=_STR:ViewInfo
		
	// Set Window Hooks
	SetWindow ParticleView#ViewControls, hook(KeyboardHook)=ParticleViewHook
	
	// Append the image showing where the particles are.
	Wave Im = $(ParticlesDFStr+"Original")
	If( WaveExists(Im) )
		DoWindow/K ImView
		NewImage/F /N=ImView /K=1 Im
		AppendImage/W=ImView FirstPerim
	EndIf
	
	DoWindow/F ParticleView
	
	ViewUpdate(ViewInfo)
End

Function ParticleViewHook(s)
	STRUCT WMWinHookStruct &s

	If(s.eventCode==11)
		switch(s.keycode)
			case 29: // Arrow Right
				ViewNextBtn("")
				break
			case 28: // Arrow Left
				ViewPrevBtn("")
				break
			case 31: // Down Arrow to Delete
				ViewDelete("")
				break
			case 32: // Space to Delete
				ViewDelete("")
		EndSwitch
	EndIf
End

Function ViewDelete(ctrlName) : ButtonControl
	String ctrlName
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
			DFREF ParticlesDF = $StringByKey("ParticlesDF",InfoStr,"{","}")
			DFREF KillDF = $StringByKey("CurrentDF",InfoStr,"{","}")
			Variable ParticleNum = ParticleNumber(StringByKey("CurrentDF",InfoStr,"{","}"))
	
	DoAlert/T=("Deleting Particle "+Num2Str(ParticleNum)+"..") 1,"Are you sure you want to delete Particle "+Num2Str(ParticleNum)+"?"
	If(V_flag!=1)
		Return 0
	EndIf
	
	Variable kill = 0
	If( ViewNextBtn("") != 0)
		If( ViewPrevBtn("") != 0)
			kill = 1
		EndIf
	EndIf
	
	KillDataFolder KillDF
		If( DataFolderRefStatus(KillDF)!=0 )
			DoAlert 0,"Could not kill the particle, it is probably open somewhere. Please close it first."
			Return -1
		EndIf
	
	Wave/SDFR=ParticlesDF /Z Volumes,OligStates,Heights,COM
		If( WaveExists(Volumes) )
			Volumes[ParticleNum]=NaN
		EndIf
		If( WaveExists(OligStates) )
			OligStates[ParticleNum]=NaN
		EndIf
		If( WaveExists(Heights) )
			Heights[ParticleNum]=NaN
		EndIf
	
	If( WaveExists(COM) )
			COM[ParticleNum][]=0
	EndIf
	
	If(kill)
		DoWindow/K ParticleView
	EndIf
	
	Return 0
End

Function ViewRange (ctrlName,varNum,varStr,varName) : SetVariableControl
	String ctrlName
	Variable varNum	// value of variable as number
	String varStr		// value of variable as string
	String varName	// name of variable
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	
	If( CmpStr(ctrlName,"XRange")==0 )
		InfoStr = ReplaceStringByKey("XRange",InfoStr,Num2Str(varNum),"{","}")
	Else
		InfoStr = ReplaceStringByKey("YRange",InfoStr,Num2Str(varNum),"{","}")
	EndIf
	
	ViewUpdate(InfoStr)
End

Function ViewInterp(ctrlName,checked) : CheckBoxControl
	String ctrlName
	Variable checked	// 1 if selected, 0 if not
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	
	InfoStr = ReplaceStringByKey("Interp",InfoStr,Num2Str(checked),"{","}")
	
	ViewUpdate(InfoStr)
End

Function ViewPerim(ctrlName,checked) : CheckBoxControl
	String ctrlName
	Variable checked	// 1 if selected, 0 if not
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	
	InfoStr = ReplaceStringByKey("Perim",InfoStr,Num2Str(checked),"{","}")
	
	ViewUpdate(InfoStr)
End

Function ViewColorRange (ctrlName,varNum,varStr,varName) : SetVariableControl
	String ctrlName
	Variable varNum	// value of variable as number
	String varStr		// value of variable as string
	String varName	// name of variable
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	
	InfoStr = ReplaceStringByKey("ColorRange",InfoStr,Num2Str(varNum),"{","}")
	
	ViewUpdate(InfoStr)
End

Function ViewColorTab (ctrlName,popNum,popStr) : PopupMenuControl
	String ctrlName
	Variable popNum	// which item is currently selected (1-based)
	String popStr		// contents of current popup item as string
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value

	InfoStr = ReplaceStringByKey("ColorTab",InfoStr,popStr,"{","}")
	ViewUpdate(InfoStr)
End

Function ViewGoTo (ctrlName,varNum,varStr,varName) : SetVariableControl
	String ctrlName
	Variable varNum	// value of variable as number
	String varStr		// value of variable as string
	String varName	// name of variable

	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	DFREF ParticlesDF = $StringByKey("ParticlesDF",InfoStr,"{","}")
	String ParticlesDFstr = StringByKey("ParticlesDF",InfoStr,"{","}")
	String CurrentDF = StringByKey("CurrentDF",InfoStr,"{","}")
		String RelativeDF = StringFromList(ItemsInList(CurrentDF,":")-1,CurrentDF,":")
	String DFList = SubfoldersList(ParticlesDF)
	
	If( varNum >= ItemsInList(DFList) )
		Return -1
	EndIf
	
	//String NewImPath = ParticlesDFstr+StringFromList(varNum,DFList)+":"+StringFromList(varNum,DFList)
	String NewImPath = ParticlesDFstr+"Particle_"+Num2Str(varNum)+":Particle_"+Num2Str(varNum)
	String NewPerimPath = ParticlesDFstr+"Particle_"+Num2Str(varNum)+":SubPixEdges_"+Num2Str(varNum) 
	Wave NewIm = $NewImPath
	Wave NewPerim = $NewPerimPath
		If( !WaveExists(NewIm) )
			Return -2
		EndIf
	
	RemoveImage/W=ParticleView /Z $RelativeDF
	RemoveImage/W=ParticleView /Z $("SubPixEdges_"+RelativeDF[9,20])
	RemoveImage/W=ImView /Z $("SubPixEdges_"+RelativeDF[9,20])
	AppendImage/W=ParticleView NewIm
	AppendImage/W=ParticleView NewPerim
	AppendImage/W=ImView NewPerim
	InfoStr = ReplaceStringByKey("CurrentDF",InfoStr,GetWavesDataFolder(NewIm,1),"{","}")
	
	ViewUpdate(InfoStr)
End

Function ViewPrevBtn (ctrlName) : ButtonControl
	String ctrlName
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	DFREF ParticlesDF = $StringByKey("ParticlesDF",InfoStr,"{","}")
	String ParticlesDFstr = StringByKey("ParticlesDF",InfoStr,"{","}")
	String CurrentDF = StringByKey("CurrentDF",InfoStr,"{","}")
		String RelativeDF = StringFromList(ItemsInList(CurrentDF,":")-1,CurrentDF,":")
	String DFList = SubfoldersList(ParticlesDF)
	
	Variable i = WhichListItem(RelativeDF,DFList), NumParticles = ItemsInList(DFList)
		If( i==-1 || i==0 )
			Return -1
		EndIf
		
	RemoveImage/W=ParticleView /Z $RelativeDF
	RemoveImage/W=ParticleView /Z $("SubPixEdges_"+Num2Str(ParticleNumber(RelativeDF)))
	RemoveImage/W=ImView /Z $("SubPixEdges_"+Num2Str(ParticleNumber(RelativeDF)))
	
	String NewImPath,NewPerimPath
	Do
		i-=1
		NewImPath = ParticlesDFstr+StringFromList(i,DFList)+":"+StringFromList(i,DFList)
		NewPerimPath = ParticlesDFstr+StringFromList(i,DFList)+":SubPixEdges_"+Num2Str(ParticleNumber(StringFromList(i,DFList)))
		Wave NewIm = $NewImPath
		Wave NewPerim = $NewPerimPath
			If( WaveExists(NewIm) )
				Break
			EndIf
	While(i<NumParticles)
		If(i==NumParticles)
			Return -2
		EndIf
	
	AppendImage/W=ParticleView NewIm
	AppendImage/W=ParticleView NewPerim
	AppendImage/W=ImView NewPerim
	InfoStr = ReplaceStringByKey("CurrentDF",InfoStr,GetWavesDataFolder(NewIm,1),"{","}")
	
	ViewUpdate(InfoStr)
	Return 0
End

Function ViewNextBtn (ctrlName) : ButtonControl
	String ctrlName
	
	ControlInfo/W=ParticleView#ViewControls Info
		String InfoStr = s_value
	DFREF ParticlesDF = $StringByKey("ParticlesDF",InfoStr,"{","}")
	String ParticlesDFstr = StringByKey("ParticlesDF",InfoStr,"{","}")
	String CurrentDF = StringByKey("CurrentDF",InfoStr,"{","}")
	String RelativeDF = StringFromList(ItemsInList(CurrentDF,":")-1,CurrentDF,":")
	String DFList = SubfoldersList(ParticlesDF)
	
	Variable i = WhichListItem(RelativeDF,DFList), NumParticles = ItemsInList(DFList)
		If( i==-1 || i==NumParticles-1 )
			Return -1
		EndIf
		
	RemoveImage/W=ParticleView /Z $RelativeDF
	RemoveImage/W=ParticleView /Z $("SubPixEdges_"+Num2Str(ParticleNumber(RelativeDF)))
	RemoveImage/W=ImView /Z $("SubPixEdges_"+Num2Str(ParticleNumber(RelativeDF)))
	
	String NewImPath, NewPerimPath
	Do
		i+=1
		NewImPath = ParticlesDFstr+StringFromList(i,DFList)+":"+StringFromList(i,DFList)
		NewPerimPath = ParticlesDFstr+StringFromList(i,DFList)+":SubPixEdges_"+Num2Str(ParticleNumber(StringFromList(i,DFList)))
		Wave NewIm = $NewImPath
		Wave NewPerim = $NewPerimPath
			If( WaveExists(NewIm) )
				Break
			EndIf
	While(i<NumParticles)
		If(i==NumParticles)
			Return -2
		EndIf
	
	AppendImage/W=ParticleView NewIm
	AppendImage/W=ParticleView NewPerim
	AppendImage/W=ImView NewPerim
	InfoStr = ReplaceStringByKey("CurrentDF",InfoStr,GetWavesDataFolder(NewIm,1),"{","}")
	
	ViewUpdate(InfoStr)
	Return 0
End

Function ViewUpdate(InfoStr)
	String InfoStr
	
	SetVariable Info win=ParticleView#ViewControls, value=_STR:InfoStr	
	String CurrentDF = StringByKey("CurrentDF",InfoStr,"{","}")
		String ImName = StringFromList(ItemsInList(CurrentDF,":")-1,CurrentDF,":")
		Variable ParticleNum = ParticleNumber(ImName)
		String PerimName = "SubPixEdges_"+Num2Str(ParticleNum)
		Wave im = $(CurrentDF+ImName)
		If( !WaveExists(im) )
			Return -1
		EndIf
	
	TitleBox ParticleName win=ParticleView#ViewControls, title="Particle "+Num2Str(ParticleNum)
	
	String ColorTab = StringByKey("ColorTab",InfoStr,"{","}")
	Variable Range = NumberByKey("ColorRange",InfoStr,"{","}")
	String theNote = Note(im)
	If( NumType(Range)!=0 || Range <=0 )
		ModifyImage/W=ParticleView $ImName ctab={*,*,$ColorTab,0}
	Else
		Variable low = NumberByKey("Background",theNote,":","\r")
		If(NumType(low)!=0)
			low = WaveMin(im)
		EndIf

		ModifyImage/W=ParticleView $ImName ctab={low,low+range,$ColorTab,0}
	EndIf
	
	ColorTab2Wave $ColorTab
	Wave Colors=:M_Colors
	ModifyGraph/W=ParticleView gbRGB=(Colors[0][0],Colors[0][1],Colors[0][2])
	KillWaves/Z Colors
	
	Variable interpo = NumberByKey("Interp",InfoStr,"{","}")
	If( interpo )
		ModifyImage/W=ParticleView $ImName interpolate=1
	Else
		ModifyImage/W=ParticleView $ImName interpolate=0
	EndIf
	
	Variable perim = NumberByKey("Perim",InfoStr,"{","}")
	If( perim )
		ModifyImage/W=ParticleView $PerimName explicit=1,eval={1,0,65535,0},eval={0,-1,-1,-1}
	Else
		ModifyImage/W=ParticleView $PerimName explicit=1,eval={1,-1	,-1,-1},eval={0,-1,-1,-1}
	EndIf
	
	// Color table for perimeter image in original image.
	ModifyImage/Z /W=ImView $PerimName explicit=1,eval={1,0,65535,0},eval={0,-1,-1,-1}
	
	Variable XRange = NumberByKey("XRange",InfoStr,"{","}")
	Variable midX = DimOffset(im,0)+DimDelta(im,0)*DimSize(im,0)/2
	If( NumType(XRange)==0 && XRange > 0)
		SetAxis/W=ParticleView bottom,midx-XRange/2,midx+XRange/2
	Else
		SetAxis/W=ParticleView /A bottom
	EndIf
	
	Variable YRange = NumberByKey("YRange",InfoStr,"{","}")
	Variable midY = DimOffset(im,1)+DimDelta(im,1)*DimSize(im,1)/2
	If( NumType(YRange)==0 && YRange > 0)
		SetAxis/W=ParticleView left,midY-YRange/2,midY+YRange/2
	Else
		SetAxis/W=ParticleView /A left
	EndIf
	
	Variable height = NumberByKey("Height",theNote,":","\r")
	Variable vol = NumberByKey("Volume",theNote,":","\r")
	ValDisplay HeightDisp win=ParticleView#ViewControls, value=_NUM:height
	ValDisplay VolDisp win=ParticleView#ViewControls, value=_NUM:vol
	
	// Resize the viewing box to keep the image aspect 1:1
	ModifyGraph/W=ParticleView height = 400*DimSize(Im,1)/DimSize(Im,0)*0.825
	
	Return 0
End

Function/S SubfoldersList(DF)
	DFREF DF
	
	If( DataFolderRefStatus(DF)!=1 )
		Return ""
	EndIf
	
	String list = ""
	Variable i,NumSubDFs = CountObjectsDFR(DF,4)
	For(i=0;i<NumSubDFs;i+=1)
		list += GetIndexedObjNameDFR(DF,4,i)+";"
	EndFor
		
	Return list
End

Function ParticleNumber(name)
	String name
	
	Variable i=StrLen(name)-1
	Do
		i-=1
	While(CmpStr(name[i],"_")!=0)
		i+=1
		
	String numStr = name[i,StrLen(name)-1]
	Return Str2Num(numStr)
End

Function FlattenButton(B_Struct) : ButtonControl
	STRUCT WMButtonAction &B_Struct
	
	If( B_Struct.eventCode==2 )	
		KillWindow/Z Image
	EndIf
	
	Return 0
End

Function FlattenSlider(S_Struct) : SliderControl
	STRUCT WMSliderAction &S_Struct
	
	If(S_Struct.eventcode==9)
		ModifyImage FLATTEN_DUP ctab={S_STruct.curval,S_Struct.curval,Grays,0},minRGB=NaN,maxRGB=(16385,28398,65535,26214)
	EndIf	
	
	Return 0
End

Function/Wave dYmap(image)
	Wave image
	
	String Name=NameOfWave(image)+"_dyMap"
	Duplicate/O image,$Name
	Wave Map=$Name
	
	Variable limQ = DimSize(image,1)-1
	Multithread Map = image[p][q] - (image[p][min(q+1,limQ)]+image[p][max(q-1,0)])/2
	
	Return Map
End

Function CleanWaveStats()
	KillVariables/Z V_npnts,V_numNans,V_numINFs,V_avg,V_sum,V_sdev,V_sem,V_rms,V_adev,V_skew,V_kurt
	KillVariables/Z V_minloc,V_min,V_maxloc,V_max,V_minRowLoc,V_maxRowLoc,V_minColLoc,V_maxColLoc
	KillVariables/Z V_minLayerLoc,V_maxLayerLoc,V_minChunkLoc,V_maxChunkLoc,V_startRow,V_endRow
	KillVariables/Z V_startCol,V_endCol,V_startLayer,V_endLayer,V_startChunk,V_endChunk
End

Function Testing(str,num)
	String str
	Variable num
	
	Print "You typed: "+str
	Print "Your number plus two is",num+2
	
End

// Kernel density estimation with an Epanechkinov kernel.
Function/Wave KernelDensity(Data,[Points,Start,Stop,Bandwidth])
	Wave Data
	Variable Points,Start,Stop,Bandwidth
	
	String Name=NameOfWave(Data)+"_Epan",theNote
	
	If( ParamIsDefault(Points) || ParamIsDefault(Start) || ParamIsDefault(Stop) || ParamIsDefault(Bandwidth) ) // Are we missing anything we need?
		Variable/C MinMax=KDE_MinMax(Data) // Make some appropriate starting guesses for the user.
			If(WaveExists($Name))
				theNote = Note($Name)
					If( NumType(NumberByKey("Bandwidth",theNote,":","\r"))==0 )
						Bandwidth = NumberByKey("Bandwidth",theNote,":","\r")
					Else
						Bandwidth=3.5*Sqrt(Variance(Data))/(NumPnts(Data)^(1/3))
					EndIf
					If( NumType(NumberByKey("Start",theNote,":","\r"))==0 )
						Start = NumberByKey("Start",theNote,":","\r")
					Else
						Start = Real(MinMax)-Bandwidth
					EndIf
					If( NumType(NumberByKey("Stop",theNote,":","\r"))==0 )
						Stop = NumberByKey("Stop",theNote,":","\r")
					Else
						Stop = Imag(MinMax)+Bandwidth
					EndIf
					If( NumType(NumberByKey("Points",theNote,":","\r"))==0 )
						Points = NumberByKey("Points",theNote,":","\r")
					Else
						Points = Max(Points,250)
					EndIf
			Else
				Bandwidth=(Imag(MinMax)-Real(MinMax) )/100
				Start=Real(MinMax)-Bandwidth
				Stop=Imag(MinMax)+Bandwidth
			EndIf
		Prompt Start,"Enter start value"
		Prompt Stop,"Enter stop value"
		Prompt Bandwidth,"Enter the bandwidth"
		Prompt Points,"Resolution? No effect on results, high res is smoother but takes longer"
			Points=Max(Points,250)
		DoPrompt "Parameters",Start,Stop,Bandwidth,Points	
			If( V_Flag==1) // Cancel clicked
					Abort
			EndIf
	EndIf
		
	If(WaveExists($Name))
		Redimension/N=(Points,0,0) $Name
	Else
		Make/N=(Points) /D $Name
	EndIf
	Wave EPDF=$Name
		EPDF[]=0
			
	Note/K EPDF
	Note EPDF,"Start:"+Num2Str(Start)
	Note EPDF,"Stop:"+Num2Str(Stop)
	Note EPDF,"Bandwidth:"+Num2Str(Bandwidth)
	Note EPDF,"Points:"+Num2Str(Points)
	Note EPDF,"SourceWave:"+GetWavesDataFolder(Data,2)
		
	SetScale/I x,Start,Stop,EPDF
	//Display/K=1  EPDF
	
	Variable i,lim=NumPnts(Data),count=NumPnts(Data)
	For(i=0;i<lim;i+=1)
		If( NumType(Data[i]) == 0)
			EPDF+=max((1-((x-Data[i])/Bandwidth)^2),0)
				//DoUpdate // Wanna see it in action??	
		Else
			count-=1
		EndIf
	EndFor
	
	EPDF/=count*4*Bandwidth/3 // Normalization such that the probability density function integrates to 1.
	
	Note EPDF,"DataPoints:"+Num2Str(count)
	
	Return EPDF
End

Function/C KDE_MinMax(Data) // Returns the min value and max value in a set of data, as a complex number for convenience
	Wave Data
	Variable/C MinMax=Cmplx(inf,-inf)
	Variable i,lim=NumPnts(Data)
	
	For(i=0;i<lim;i+=1)
		If( NumType(Data[i])==0)
		
			If( Data[i]<Real(MinMax))
				MinMax=Cmplx( Data[i] , Imag(MinMax) )
			EndIf
			If( Data[i]>Imag(MinMax))
				MinMax=Cmplx( Real(MinMax) , Data[i] )
			EndIf
		EndIf
	EndFor

	Return MinMax
End