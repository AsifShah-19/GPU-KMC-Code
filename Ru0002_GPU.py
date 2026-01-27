# KMC_Ru0002_CUDA_Fused_Kernel

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import cuda
import math as mh
import time

#Physical parameters
a = np.float64(271e-12)                                                     # Surface distance parameters
xD = np.float64(a*mh.cos(mh.radians(60)))                                   # Surface distance parameters
yD = np.float64(a*mh.sin(mh.radians(60)))                                   # Surface distance parameters
kB = np.float64(8.617e-5)                                                   # Boltzmann constant (eV/K)
T = 923                                                                     # Temperature (K)
PF = 1e13                                                                   # Attempt frequency (1/s)
G = 4                                                                       # Geometric parameter (e.g., 2D(Hexagon) has 4(6) jumps directions)

#Energy and rates
Ea = np.array([1.73,1.73,1.73,1.73,1.73,1.73], dtype=np.float64)              # DFT-LCAO GGA level CINEB energies from a reference Ru(0002) atom to neighboring six Ru(0002) atoms
ka = PF*np.exp(-Ea/(kB*T))                                                    # Rate constant
ktot = sum(ka)                                                                # Total rate constant
Normka = ka/ktot                                                              # Normalized rate constants
CumProb_host = np.cumsum(Normka)                                              # Cumulative probabilites on CPU
CumProb = cuda.to_device(CumProb_host)                                        # Cumulative probabilites copied to GPU

#Jump vectors
V0_host=np.array([[xD,yD],[a,0],[xD,-yD],[-xD,-yD],[-a,0],[-xD,yD]], dtype=np.float64)   # Jump vectors on CPU
V0 = cuda.to_device(V0_host)                                                             # Jump vectors copied to GPU

#KMC parameters
KMCSteps = 1000000                                                          # KMC steps per ensemble
Ensem = 100000                                                              # Total number of ensembles = Threads on GPU
threads_per_block = 128                                                     # Dividing threads on GPU into blocks (Limit - 1024 threads/block)
sum_MSD = cuda.device_array(KMCSteps, dtype=np.float64)                     # Initializing array on GPU where MSD per step from different ensembles/threads will be added
sum_Time = cuda.device_array(KMCSteps, dtype=np.float64)                    # Initializing array on GPU where timesteps from different ensembles/threads will be added
d_seeds = cuda.device_array(Ensem, dtype=np.uint32)                         # Initializing array on GPU where random seeds for each ensemble/thread will be stored

@cuda.jit                                                                   # Translates and compiles python code into CUDA kernels for GPU execution
def kmc_fused_kernel(V0, ktot, KMCSteps, d_seeds, sum_MSD, sum_Time, CumProb):
    i = cuda.grid(1)                                                        # Obtaining global thread index
    if i >= d_seeds.size:                                                   # Ensuring threads dont exceed total number of ensembles
        return
    seed = d_seeds[i]                                                       # Assigning random number for each thread based on its global index i

    PRX = seed * np.uint32(123457) % np.uint32(199)
    PRY = seed * np.uint32(123458) % np.uint32(199)
    x0 = float(PRX) * xD                                                    # Random initialization of initial position of vacancy (x coordinate) within a 198 x 198 supercell
    y0 = float(PRY) * yD                                                    # Random initialization of initial position of vacancy (y coordinate) within a 198 x 198 supercell
    x = x0
    y = y0
    t = 0.0                                                                 # Initializing time
    
    for step in range(KMCSteps):
        #Choosing a random number for advancing position of vacancy
        seed ^= (seed << np.uint32(13))                                     # XORing seed to create more randomness
        seed ^= (seed >> np.uint32(17))
        seed ^= (seed << np.uint32(5))
        r1 = float(seed & np.uint32(0xFFFFFFFF)) / 4294967296.0             # This is a random number between [0,1)

        for i in range(CumProb.size):                                       # Choosing a jump vector based on random number r1
            if r1 < CumProb[i]:
                index = i
                break

        x += V0[index, 0]                                                   # Advancing x position of vacancy based on random number r1
        y += V0[index, 1]                                                   # Advancing y position of vacancy based on random number r1
        
        #Choosing a random number for advancing time
        seed ^= (seed << np.uint32(13))                                     # XORing seed to create more randomness
        seed ^= (seed >> np.uint32(17))
        seed ^= (seed << np.uint32(5))
        r2 = max(float(seed & np.uint32(0xFFFFFFFF)) / 4294967296.0, 1e-10) # This is a random number between (0,1)

        t += -mh.log(r2) / ktot                                             # Advancing time based on random number r2

        #Calculating Instantaneous Mean Square Displacement (MSD)
        dx = x - x0
        dy = y - y0
        msd = dx*dx + dy*dy
        #Adding up MSD and timesteps from all ensembles/threads on GPU
        cuda.atomic.add(sum_MSD, step, msd)
        cuda.atomic.add(sum_Time, step, t)


def main():
    zero_arr = np.zeros(KMCSteps, dtype=np.float64)                         # Defining an array on CPU
    sum_MSD.copy_to_device(zero_arr)                                        # Copying CPU array to GPU array
    sum_Time.copy_to_device(zero_arr)                                       # Copying CPU array to GPU array

    host_seeds = np.random.randint(0, 2**32 - 1, size=Ensem, dtype=np.uint32) # Generating random seed array on CPU
    d_seeds.copy_to_device(host_seeds)                                      # Copying random seed array to GPU

    blocks = (Ensem + threads_per_block - 1) // threads_per_block           # Calculating number of blocks - Can be more than ensemble size, therefore, its limited inside KMC_fused_kernel function

    start = time.time()                                                     # Starting timer
    kmc_fused_kernel[blocks, threads_per_block](                            # Launching Fused Kernel KMC
        V0, ktot, KMCSteps, d_seeds, sum_MSD, sum_Time, CumProb
    )
    cuda.synchronize()
    end = time.time()                                                       # Stopping timer

    TotalMSD = sum_MSD.copy_to_host() / Ensem                               # Averaging MSD and copying it back to CPU
    TotalTime = sum_Time.copy_to_host() / Ensem                             # Averaging time and copying it back to CPU

    # Saving into excel
    data = {'Time': TotalTime, 'MSD': TotalMSD}
    df = pd.DataFrame(data)
    df.to_excel('/home/asif/Ru0002_GPU.xlsx', index=False)
    
    slope, intercept = np.polyfit(TotalTime, TotalMSD, 1)
    D = slope / G
    print(f"GPU fused kernel wall‑time = {end - start:.8f} s")
    Datom=D*np.exp(-(1.947433-1.5*kB*T)/(kB*T))                             # DFT-LCAO GGA level based vacancy formation energy Ef = -1.947433 eV, Vacancy formation entropy = 1.5kB
    print("Vacancy Diffusion = ", D, " m^2/s \nAtomic Diffusion = ", Datom, " m^2/s")

if __name__ == "__main__":
    main()

