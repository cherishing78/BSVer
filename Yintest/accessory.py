import numpy as np

def Initial_diag(dim):
	conv=np.diag(np.random.rand(dim))
	return conv

def Convergence(matrix):
	delta=(np.abs(matrix).max(axis=0)).max(axis=0)
	return delta

def Mean(dataset):
	mean=np.mean(dataset,axis=0)
	return mean

def Train(trainingset,label):
	(imagenum,dim)=trainingset.shape
	#Each column vector stands for a image.
	dataset=np.transpose(trainingset)
	label.shape=(-1,)
	peoplenum=label[-1]
	m=np.zeros(peoplenum,dtype=np.uint16)
	#m[i] stands for the num of images the i th people has.
	#The elements in label start with 1.
	for i in label:
		m[i-1]+=1
	#Delete the repetitive elements and get the m_set list.	
	m_set=set(list(m))
	m_max=max(m_set)
	print '------ m_set Accomplished  ------'
	print m_set
	#Initialization
	Su=Initial_diag(dim)
	Se=Initial_diag(dim)
	print '------ Initialization Accomplished ------'
	#Iteration
	epsilon=1e-4
	Delta_Su=Su
	Delta_Se=Se
	Iter=0
	Delta=max(Convergence(Delta_Su),Convergence(Delta_Se))
	print '------ Training Process ------'
	while Delta>epsilon:
		print '------ Delta=%f in %dth Iteration------'%(Delta,Iter)
		#Compute the F and all kinds of G in each iteration time.
		F=np.linalg.pinv(Se)
		#In case there is no people has m[k] images.
		G_class=[0 for i in range(m_max)]
		for i in range(1,m_max+1):
			if i in m_set:
				#Compute various G in advance for the sake of convenience.
				G_class[i-1]=-np.dot(np.linalg.pinv((i+1)*Su+Se),np.dot(Su,F))
				print '------ G_class[%d] Accopmlished in the %dth Iteration ------'%(i-1,Iter)
		#Compute u[i] for each person and e[i,j] for each image.
		#Initialize the Pointer of each person's images.
		m_index=0
		Su_new=0
		Se_new=0
		print '------ Compute the Su_new an Se_new in %dth Iteration'%Iter
		for i in range(peoplenum):
			u=0
			e=0
			#Compute the constant term for e[i,j].
			constant=0
			for j in range(m_index,m_index+m[i]):
				constant+=np.dot(Se,np.dot(G_class[m[i]-1],dataset[:,j]))
			#Compute the Su_new and Se_new
			for j in range(m_index,m_index+m[i]):
				u+=np.dot(Su,np.dot((F+(m[i]+1)*G_class[m[i]-1]),dataset[:,j]))
				eij=np.dot(Se,dataset[:,j])+constant
				Se_new+=np.dot(eij,np.transpose(eij))/m[i]/peoplenum
			Su_new+=np.dot(u,np.transpose(u))/peoplenum				
			#Pointer move on.
			m_index+=m[i]
		Delta_Su=Su_new.__sub__(Su)
		Delta_Se=Se_new.__sub__(Se)
		Delta=max(Convergence(Delta_Su),Convergence(Delta_Se))
		Su=Su_new
		Se=Se_new	
		print '------ %dth iteration accomlished ------'%Iter	
		Iter+=1
		if Iter>10:
			break
	#Get the matrix in need.
	F=np.linalg.pinv(Se)
	#Save the memory.
	if 1 not in m_set:
		G_class[0]=-np.dot(np.dot(np.linalg.pinv(2*Su+Se),Su),F)
	A=np.linalg.pinv(Su+Se)-F-G_class[0]
	return A,G

def Verify(A,G,x1,x2):
	x1.shape=(-1,1)
	x2.shape=(-1,1)
	ratio=np.dot(np.dot(np.transpose(x1),A),x1)+np.dot(np.dot(np.transpose(x2),A),x2)-2*np.dot(np.dot(np.transpose(x1),G),x2)
	return ratio

