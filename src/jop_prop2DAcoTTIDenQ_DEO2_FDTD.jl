function JetProp2DAcoTTIDenQ_DEO2_FDTD(;
        v = Float32[],
        b = Float32[],
        ϵ = Float32[],
        η = Float32[],
        θ = Float32[],
        f = 0.85,
        srcfieldfile = joinpath(tempdir(), "field-$(uuid4()).bin"),
        comptype = nothing,
        compscale = 1e-2,
        isinterior = false,
        nz_subcube = 32,
        nx_subcube = 32,
        sz = 0.0,
        sx = 0.0,
        st = 0.0,
        interpmethod = :hicks,
        rz = [0.0],
        rx = [0.0],
        z0 = 0.0,
        x0 = 0.0,
        dz = 10.0,
        dx = 10.0,
        freqQ = 5.0,
        qMin = 0.1,
        qInterior = 100.0,
        padz = 0.0,
        padx = 0.0,
        dtmod,
        dtrec,
        ntrec,
        nbz_cache = 512,
        nbx_cache = 8,
        nsponge = 50,
        wavelet = WaveletCausalRicker(f=5.0),
        freesurface = false,
        imgcondition = "standard",
        RTM_weight = 0.5, 
        nthreads = Sys.CPU_THREADS,
        reportinterval = 500)
    # active and passive earth model properties.  The active set is in the model-space
    active_modelset = Dict{String,Int}()
    passive_modelset = Dict{String,Array{Float32,2}}()
    i = 1
    for (n,x) in (("v",v), ("b",b), ("ϵ",ϵ), ("η",η))
        if length(x) > 0
            passive_modelset[n] = x
        else
            active_modelset[n] = i # so that m[:,:,i] corresponds to property n∈("v","b","ϵ","η")
            i += 1
        end
    end
    passive_modelset["sinθ"] = isempty(θ) ? zeros(Float32, size(values(active_modelset)[1])) : sin.(convert(Array{Float32}, θ));
    passive_modelset["cosθ"] = isempty(θ) ? ones(Float32, size(values(active_modelset)[1])) : cos.(convert(Array{Float32}, θ));
    @assert length(active_modelset) > 0

    # active and passive wavefields (an active wavefield is serialized to disk)
    active_modelset_keys = keys(active_modelset)
    local active_wavefields, modeltype
    if "v" ∈ active_modelset_keys && "b" ∉ active_modelset_keys && "ϵ" ∈ active_modelset_keys && "η" ∈ active_modelset_keys
        active_wavefields = ["pold","mold","pspace","mspace"]
        modeltype = WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD_Model_VEA()
    elseif "v" ∈ active_modelset_keys && "b" ∉ active_modelset_keys && "ϵ" ∉ active_modelset_keys && "η" ∉ active_modelset_keys
        active_wavefields = ["pspace","mspace"]
        modeltype = WaveFD.Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V()
    else
        error("unsupported model-space")
    end

    # dtrec must be integer multiple of dtmod
    @assert abs(dtrec - dtmod*round(dtrec/dtmod,RoundNearest)) < eps(Float32)

    # source location (sz,sx,st)
    sz,sx,st = map(val->Float64[val...], (sz,sx,st))

    # domain and range
    dom = JetSpace(Float32, size(first(values(passive_modelset)))..., length(active_modelset))
    rng = JetSpace(Float32, ntrec, length(rz))

    # type conversions
    z0, x0, padz, padx = map(val->Float64(val), (z0, x0, padz, padx)) # used for computing integer grid locations for source injection
    rz, rx = map(val->convert(Array{Float64}, val), (rz, rx))
    dz, dx, dtmod, dtrec, f, freqQ, qMin, qInterior = map(val->Float32(val), (dz, dx, dtmod, dtrec, f, freqQ, qMin, qInterior)) # used in F.D. calculations
    @assert length(rz) == length(rx)
    if isa(wavelet, Array) == true
        wavelet = convert(Array{Float32}, wavelet)
    end

    # ginsu view of earth model... ginsu is aware of the sponge
    nsponge_top = freesurface ? 0 : nsponge
    padz_top = freesurface ? 0.0 : padz

    ginsu = Ginsu((z0,x0), (dz,dx), size(b), (sz,sx), (rz,rx), ((padz_top,padz),(padx,padx)), ((nsponge_top,nsponge),(nsponge,nsponge)), T=Float32)

    # if srcfieldfile is specified make sure its containing folder exists
    if srcfieldfile != ""
        srcfieldpath = split(srcfieldfile, "/")[1:end-1]
        if length(srcfieldpath) > 0
            srcfieldpath = join(srcfieldpath, "/")
            mkpath(srcfieldpath)
        end
    end

    # compression for nonlinear source wavefields
    C = WaveFD.comptype(comptype, Float32)[1]
    compressor = Dict{String,WaveFD.Compressor{Float32,Float32,C,2}}()

    # we need to serialize for the data, but not for the linearization
    _active_wavefields = "pold" ∈ active_wavefields ? active_wavefields : [active_wavefields;"pold"]

    for active_wavefield in _active_wavefields
        compressor[active_wavefield] = WaveFD.Compressor(Float32, Float32, C, size(ginsu,interior=isinterior), 
            (nz_subcube,nx_subcube), compscale, ntrec, isinterior)
    end

    # imaging condition 
    icdict = Dict(
        lowercase("standard") => WaveFD.ImagingConditionStandard(),
        lowercase("FWI") => WaveFD.ImagingConditionWaveFieldSeparationFWI(),
        lowercase("RTM") => WaveFD.ImagingConditionWaveFieldSeparationRTM(),
        lowercase("MIX") => WaveFD.ImagingConditionWaveFieldSeparationMIX())

    if lowercase(imgcondition) ∉ keys(icdict)
        error("Supplied imaging condition 'imgcondition' is not in [standard, FWI, RTM, MIX]")
    end
        
    # construct:
    Jet(
        dom = dom,
        rng = rng,
        f! = JopProp2DAcoTTIDenQ_DEO2_FDTD_f!,
        df! = JopProp2DAcoTTIDenQ_DEO2_FDTD_df!,
        df′! = JopProp2DAcoTTIDenQ_DEO2_FDTD_df′!,
        s = (
            modeltype = modeltype,
            active_modelset = active_modelset,
            passive_modelset = passive_modelset,
            active_wavefields = active_wavefields,
            f = Float32(f),
            srcfieldfile = srcfieldfile,
            srcfieldhost = Ref(""),
            chksum = Ref(zero(UInt32)),
            compressor = compressor,
            isinterior = isinterior,
            sz = sz,
            sx = sx,
            st = st,
            interpmethod = interpmethod,
            rz = rz,
            rx = rx,
            z0 = z0,
            x0 = x0,
            dz = dz,
            dx = dx,
            freqQ = freqQ,
            qMin = qMin,
            qInterior = qInterior,
            ginsu = ginsu,
            nsponge = nsponge,
            dtmod = dtmod,
            dtrec = dtrec,
            ntrec = ntrec,
            nbz_cache = nbz_cache,
            nbx_cache = nbx_cache,
            wavelet = wavelet,
            freesurface = freesurface,
            imgcondition = get(icdict, lowercase(imgcondition), WaveFD.ImagingConditionStandard()),
            RTM_weight = RTM_weight,
            nthreads = nthreads,
            reportinterval = reportinterval,
            stats = Dict{String,Float64}("MCells/s"=>0.0, "%io"=>0.0, "%inject/extract"=>0.0, "%imaging"=>0.0)))
end

@doc """
    JopNlProp2DAcoTTIDenQ_DEO2_FDTD(; kwargs...)
    JopLnProp2DAcoTTIDenQ_DEO2_FDTD(; m₀, kwargs...) 

Create a `Jets` nonlinear or linearized operator for 2D pseudo- visco-acoustic, tilted transverse isotropy, 
variable density modeling.

# Model Parameters
This propagator operates with five model parameters, as shown in the table below.

| Parameter | Description                                       |
|:---------:|:--------------------------------------------------|
| `v`       | P wave velocity                                   |
| `ϵ`       | Modified Thomsen's weak anisotropy parameter      |
| `η`       | Modified Alkhalifah's weak anisotropy parameter η |
| `b`       | buoyancy (reciprocal density)                     |
| `θ`       | symmetry axis tilt angle from vertical (radians)  |

## Pseudo-acoustic approximation 
With the pseudo-acoustic approximation we employ, the transformation to self-adjoint form requires a 
parameter `f` representing the average ratio of shear-wave to p-wave velocity, and a modfication 
to Alkhalifah's weak anisotropy parameter `η`. We show formulas for these two parameters below. 
The parameter `f` is specified as a scalar for the entire model, with default value 0.85, implying
a shear velocity around 38% of the p wave velocity.
* `η = sqrt[2(ϵ-δ)/(f+2ϵ)]`
* `f = 1 - Vₛ² / Vₚ²`

For more information about the pseudo-acoustic approximation we employ please see 
https://library.seg.org/doi/10.1190/segam2016-13878451.1. 

## Active and passive model parameters 
There are two modes of operation that define different sets of **active** and **passive** parameters, 
*velocity only* and *velocity and anisotropy*. 

An **active** parameter can be inverted for using the Jacobian linearized operator machinery,  
and a **passive** parameter is constant. The two modes are:

| Mode                    | Active Parameters | Passive Parameters    |
|:------------------------|:------------------|:----------------------|
| velocity only           | [`v`]             | [`ϵ`, `η`, `b`, `θ` ] |
| velocity and anisotropy | [`v`, `ϵ`, `η`]   | [`b`, `θ` ]           |

To make a parameter **passive**, you pass the array for that parameter in the constructor for the 
operator, implying that it is in state and does not change with the action of the operator. 
Parameters that are *not* passed to the constructor are assumed to be **active**, and will be part 
of the model that the operator acts on, stored as a 3D array with the following indices: 
`[Z coord][X coord][Active Parameter]`

# Examples

## Model and acquisition geometry setup
1. load modules Jets, WaveFD, and JetPackWaveFD
1. set up the model discretization, coordinate size and spacing
1. set up the acquisition geometry, including the time discretization, locations for source and receivers, and the source wavelet
1. create constant models for `v`, `ϵ`, `η`, `b`, `θ`
```
using Jets, WaveFD, JetPackWaveFD
nz,nx = 200, 100                      # spatial discretization size
dz,dx = 20.0, 20.0                    # spatial discretization sampling
ntrec = 1101                          # number of temporal samples in recorded data
dtrec = 0.0100                        # sample rate for recorded data
dtmod = 0.0025                        # sample rate for modeled data
wavelet = WaveletCausalRicker(f=5.0)  # type of wavelet to use (source signature) 
sz = dz                               # source Z location 
sx = dx*(nx/2)                        # source X location 
rx = dx*[0:0.5:nx-1;];                # array of receiver X locations
rz = 2*dz*ones(length(0:0.5:nx-1));   # array of receiver Z locations
b = ones(Float32, nz, nx);            # constant buoyancy
ϵ = 0.1 * ones(Float32, nz, nx);      # constant epsilon
η = 0.2 * ones(Float32, nz, nx);      # constant eta
θ = (π/8) * ones(Float32, nz, nx);    # constant tilt angle
```

## Construct and apply the nonlinear operator (`v` is active; `ϵ, η, b, θ` are passive)
1. create the nonlinear operator `F`
1. create the model vector and set parameter `v`
1. perform nonlinear forward modeling with model `m₀` and return the resulting modeled data in `d`
```
F = JopNlProp2DAcoTTIDenQ_DEO2_FDTD(; b = b, ϵ = ϵ, η = η, θ = θ, ntrec = ntrec, dtrec = dtrec, 
    dtmod = dtmod, dz = dz, dx = dx, wavelet = wavelet, sz = sz, sx = sx, rz = rz, rx = rx)
m₀ = zeros(domain(F));
m₀[:,:,1] .= 1500;
d = F*m₀;              # forward nonlinear op
```

## Construct and apply the nonlinear operator (`v, ϵ, η` are active; `b, θ` are passive)
1. create the nonlinear operator `F`
1. create the model vector and set parameters `v, ϵ, η`
1. perform nonlinear forward modeling with model `m₀` and return the resulting modeled data in `d`
```
F = JopNlProp2DAcoTTIDenQ_DEO2_FDTD(; b = b, θ = θ, ntrec = ntrec, dtrec = dtrec, dtmod = dtmod, 
    dz = dz, dx = dx, wavelet = wavelet, sz = sz, sx = sx, rz = rz, rx = rx)
m₀ = zeros(domain(F));
m₀[:,:,1] .= 1500;
m₀[:,:,2] .= 0.1;
m₀[:,:,3] .= 0.2;
d = F*m₀;              # forward nonlinear op
```

## Construct and apply the linearized Jacobian operator (method 1)
For this example we assume in the model that `v, ϵ, η` are active parameters, and `b, θ` are passive.
1. create the nonlinear operator `F` directly by construction of a jet. 
1. create the model vector and set parameters `v, ϵ, η`
1. create the Jacobian operator `J` by linearization of `F` at point `m₀` 
1. create a random model perturbation vector `δm`.
1. perform linearized forward (Born) modeling on the model perturbation vector `δm` and 
    return the resulting data perturbation in `δd`.
1. perform linearized adjoint (Born) migration on the data perturbation vector `δd` and 
    return the resulting model perturbation in `δm`. 

Note that the Jacobian operators `J` and `J'` require the serialized nonlinear forward wavefield, 
and this is generated automatically whenever required. If you watch the logging to standard out
from this example, you will first see the finite difference evolution for the nonlinear forward, 
followed by the linearized forward, and finally the linearized adjoint.
```
F = JopNlProp2DAcoTTIDenQ_DEO2_FDTD(; b = b, θ = θ, ntrec = ntrec, dz = dz, dx = dx, 
    dtrec = dtrec, dtmod = dtmod, wavelet = wavelet, sz = sz, sx = sx, rz = rz, rx = rx)
m₀ = zeros(domain(F));
m₀[:,:,1] .= 1500;
m₀[:,:,2] .= 0.1;
m₀[:,:,3] .= 0.2;
J = jacobian(F, m₀)
δm = rand(domain(J));
δd = J*δm;             # forward linearized op
δm = J'*δd;            # adjoint linearized op
```

## Construct and apply the linearized Jacobian operator (method 2)
For this example we assume in the model that `v, ϵ, η` are active parameters, and `b, θ` are passive.
1. create the model vector and set parameters `v, ϵ, η`
1. create the Jacobian operator `J` at point `m₀` directly by construction of a jet.
1. create a random model perturbation vector `δm`.
1. perform linearized forward (Born) modeling on the model perturbation vector `δm` and 
    return the resulting data perturbation in `δd`.
1. perform linearized adjoint (Born) migration on the data perturbation vector `δd` and 
    return the resulting model perturbation in `δm`. 
```
m₀ = zeros(Float32, nz, nx, 3);
m₀[:,:,1] .= 1500;
m₀[:,:,2] .= 0.1;
m₀[:,:,3] .= 0.2;
J = JopLnProp2DAcoTTIDenQ_DEO2_FDTD(; m₀ = m₀, b = b, θ = θ, ntrec = ntrec, dz = dz, dx = dx, 
    dtrec = dtrec, dtmod = dtmod, wavelet = wavelet, sz = sz, sx = sx, rz = rz, rx = rx)
δm = rand(domain(J));
δd = J*δm;             # forward linearized op
δm = J'*δd;            # adjoint linearized op
```

# Required Parameters
* `m₀` the point at which the jet is linearized. Note this argument is required in the constuctor for  
    `JopLnProp2DAcoTTIDenQ_DEO2_FDTD` but not `JopNlProp2DAcoTTIDenQ_DEO2_FDTD`. This constuctor is shown  
    in the last example above. Please note that you must consider which parameters are active and passive,
    per the discussion on model parameters above and examples.
* `dtmod` The sample rate for the modeled data. You can establish a lower bound for the modeling sample rate  
    with the following expression: `dt = 0.75 * 0.38 * max(dx,dz) / maximum(v)`. Note that the usual  
    Courant–Friedrichs–Lewy condition (CFL condition) for stability in finite difference modeling is  
    modified *heuristically* to include the impact of the visco-acoustic implementation of this operator,  
    requiring a 25% smaller smaple rate.
* `dtrec` The number of time samples in the recorded data. Note requirement that `dtrec` 
    be an even multiple of `dtmod`.
* `ntrec` The number of time samples in the recorded data. Note that the number of samples in the modeled  
    data is determined from `ntrec`, `dtrec`, and `dtmod`.

# Optional Parameters
Defaults for arguments are shown inside square brackets.

* `b [ones(Float32,0,0)]` the buoyancy (reciprocal density) array.
* `θ [zeros(Float32,0,0)]` the symmetry axies tilt angle in radians. Assumed to be zero if not 
    specified. θ must be an array the same size as buoyancy `b`. 
* `f [0.85]` See the discussion above regarding the **Pseudo-acoustic approximation** used here
* `srcfieldfile [joinpath(tempdir(), "field-$(uuid4()).bin")]` the full path to a scratch file used for 
    the serializationof the compressed nonlinear source wavefield. 
* `comptype [nothing]` the type of compression to use for the serialization of
    the nonlinear source wavefield. The type of compression must be one of:
    * `nothing` - no compression.
    * `Float32` - if wavefield is `Float64`, then do a simple conversion to Float32.
    * `UInt32` - compression using CvxCompress (windowing + 2D wavelet transform + 
        thresholding + quantization + run-length-encoding).
* `compscale [1e-2]` determines the thresholding for the compression of the nonlinear source 
    wavefield prior to serialization. Larger values mean more aggressive compression. You can 
    likely increase compscale to 1.0 before you start to notice differences in output from 
    Jacobian operations. 
* `nz_subcube, nx_subcube [32]` The Z and X sizes of windows used for compression 
    of the nonlinear source wavefield with `CvxCompress`. Note the requirement
    `[8 <= n*_subcube <=256]`.
* `isinterior [false]` boolean flag that indicates how the nonlinear source wavefield is
    saved. For large models, operation will be faster with `isinterior = true`, but 
    the linearization correctness test may fail.
    * `true` the entire model including absorbing boundaries is serialized and deserialized
    * `false` the interior part of the model excluding absorbing boundaries is serialized 
        and deserialized
* `sz [0.0]` Array of source Z coordinate. Note that if multiple sources are provided, 
    they will will be injected simultaneously during finite difference evolution. 
* `sx [0.0]` Array of source X coordinate.
* `st [0.0]` Array of source delay times.
* `interpmethod [:hicks]` Type of physical interpolation for sources and receivers. For locations 
    that are not on the physical grid coordinates, interpolation is used either to inject or extract
    information. `interpmethod` must be one of:
    * `:hicks` Hicks 2D sinc interpolation (up to 8x8 nonzero points per location)
    * `:linear` bilinear interpolation (up to 2x2 nonzero points per location)
* `rz [[0.0]]` 2D array of receiver Z coordinates
* `rx [[0.0]]` 2D array of receiver Z coordinates
* `z0 [0.0]` Origin of physical coordinates for the Z dimension.
* `x0 [0.0]` Origin of physical coordinates for the X dimension.
* `dz [10.0]` Spacing of physical coordinates in the Z dimension.
* `dx [10.0]` Spacing of physical coordinates in the X dimension.
* `freqQ [5.0]` The center frequency for the Maxwell body approximation to dissipation only attenuation.
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `qMin [0.1]` The minimum value for Qp at the boundary of the model used in our Maxwell body approximation
    to dissipation only attenuation. This is not a physically meaningful value for Qp, as we use the 
    attenuation to implement absorbing boundary conditions and eliminate outgoing waves on the 
    boundaries of the computational domain. 
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `qInterior [100.0]` the value for Qp in the interior of the model used in our Maxwell body approximation
    to dissipation only attenuation. This is the value for Qp away from the absorbing boundaries and is 
    a physically meaningful value.
    Please see `JetPackWaveFD` package documentation for more information concerning the attenuation model.   
* `padz, padx [0.0], [0.0]` - apply extra padding to the survey determined aperture in `Ginsu`.
    Please see `Ginsu` for more information. 
* `nbz_cache, nbx_cache [512], [8]` The size of cache blocks in the Z and X dimensions. 
    In general the cache block in the Z (fast) dimension should be ≥ the entire size of that dimension, 
    and the cache block size in the slower dimension is generally small in order to allow the entire
    block to fit in cache. 
* `nsponge [50]` The number of grid cells to use for the absorbing boundary. For high fidelity modeling
    this should be > 60 grid points, but can be significantly smaller for some use cases like low frequency 
    full waveform inversion. 
* `wavelet [WaveletCausalRicker(f=5.0)]` The source wavelet, can be specified as either a Wavelet type 
    or an array.
* `freesurface [false]` Determines if a free surface (`true`) or absorbing (`false`) top boundary condition
    is applied.
* `imgcondition` ["standard"] Selects the type of imaging condition used. Choose from "standard", "FWI", 
    and "RTM". "FWI" and "RTM" will perform Kz wavenumber filtering prior to the imaging condition
    in order to promote long wavelengths (for FWI), or remove long wavelength backscattered energy (for 
    RTM). "MIX" mixes the FWI imaging condition using the parameter RTM_weight. Note the true adjoint 
    only exists for "standard" imaging condition currently.
* `RTM_weight [0.5]` determines the balance of short wavelengths and long wavelengths in the imaging condition. 
    A value of 0.0 is equivalent to the FWI imaging condition, a value of 0.5 is equivalent to the standard
    imaging condtion, and a value of 1.0 is equivalent to the RTM imaging condition. 
* `nthreads [Sys.CPU_THREADS]` The number of threads to use for OpenMP parallelization of the modeling.
* `reportinterval [500]` The interval at which information about the propagtion is logged.

See also: `Ginsu`, `WaveletSine`, `WaveletRicker`, `WaveletMinPhaseRicker`, `WaveletDerivRicker`, 
    `WaveletCausalRicker`, `WaveletOrmsby`, `WaveletMinPhaseOrmsby` 
"""
JopNlProp2DAcoTTIDenQ_DEO2_FDTD(;kwargs...) = JopNl(JetProp2DAcoTTIDenQ_DEO2_FDTD(;kwargs...))

@doc (@doc JopNlProp2DAcoTTIDenQ_DEO2_FDTD)
JopLnProp2DAcoTTIDenQ_DEO2_FDTD(; m₀, kwargs...) = JopLn(JetProp2DAcoTTIDenQ_DEO2_FDTD(;kwargs...), m₀)

export JopNlProp2DAcoTTIDenQ_DEO2_FDTD
export JopLnProp2DAcoTTIDenQ_DEO2_FDTD

function JopProp2DAcoTTIDenQ_DEO2_FDTD_nonlinearforward!(d::AbstractArray, m::AbstractArray; kwargs...)
    # make propagator
    nz_ginsu,nx_ginsu = size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu = origin(kwargs[:ginsu])
    p = Prop2DAcoTTIDenQ_DEO2_FDTD(
        nz = nz_ginsu,
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache],
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz],
        dx = kwargs[:dx],
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ],
        qMin = kwargs[:qMin],
        qInterior = kwargs[:qInterior])

    # wave-fields
    wavefields = Dict("pcur"=>WaveFD.PCur(p), "pold"=>WaveFD.POld(p), "pspace"=>WaveFD.PSpace(p), "mcur"=>WaveFD.MCur(p), "mold"=>WaveFD.MOld(p), "mspace"=>WaveFD.MSpace(p))

    # earth model
    model_ginsu = Dict("v"=>WaveFD.V(p), "b"=>WaveFD.B(p), "ϵ"=>WaveFD.Eps(p), "η"=>WaveFD.Eta(p), "f"=>WaveFD.F(p), 
        "cosθ"=>WaveFD.CosTheta(p), "sinθ"=>WaveFD.SinTheta(p))

    # ginsu'd earth model (active-set)
    for prop in keys(kwargs[:active_modelset])
        sub!(model_ginsu[prop], kwargs[:ginsu], @view(m[:,:,kwargs[:active_modelset][prop]]), extend=true)
    end
    for prop in keys(kwargs[:passive_modelset])
        sub!(model_ginsu[prop], kwargs[:ginsu], kwargs[:passive_modelset][prop], extend=true)
    end
    model_ginsu["f"] .= kwargs[:f]
    ginsu_interior_range = interior(kwargs[:ginsu])

    # we need to serialize "pold" for the data but, not (necessarily) for the linearization
    active_wavefields = "pold" ∈ kwargs[:active_wavefields] ? kwargs[:active_wavefields] : [kwargs[:active_wavefields]; "pold"]

    it0, ntmod_wav = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:st], kwargs[:ntrec])

    # source wavelet for injection, one for each source location
    wavelet_realization = realizewavelet(kwargs[:wavelet], kwargs[:sz], kwargs[:sx], kwargs[:st], kwargs[:dtmod], ntmod_wav)

    # Get source and receiver interpolation coefficients
    local points_sou
    if kwargs[:interpmethod] == :hicks
        points_sou = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:sz], kwargs[:sx])
    else
        points_sou = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:sz], kwargs[:sx])
    end

    # Source blocking after scaled
    points_sou_scaled = Vector{WaveFD.SourcePoint32}(undef, length(points_sou))
    for i in eachindex(points_sou_scaled)
        ju = points_sou[i].iu
        c_sou = points_sou[i].c * model_ginsu["b"][ju] / kwargs[:dx] / kwargs[:dz]
        c_sou_scaled = c_sou * kwargs[:dtmod]^2 * model_ginsu["v"][ju]^2 / model_ginsu["b"][ju]
        points_sou[i] = WaveFD.SourcePoint32(ju, points_sou[i].ir, c_sou)
        points_sou_scaled[i] = WaveFD.SourcePoint32(ju, points_sou[i].ir, c_sou_scaled)
    end
    blks_sou = WaveFD.source_blocking(points_sou, kwargs[:nthreads])
    blks_sou_scaled = WaveFD.source_blocking(points_sou_scaled, kwargs[:nthreads])

    local blks_rec
    if length(d) > 0
        local points_rec
        if kwargs[:interpmethod] == :hicks
            points_rec = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
        else
            points_rec = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
        end
        blks_rec = WaveFD.receiver_blocking(points_rec, kwargs[:nthreads])
    end

    # disk-file for source-field serialization
    iofield = Dict{String,IOStream}()
    if kwargs[:srcfieldfile] != ""
        for active_wavefield in active_wavefields
            filename = "$(kwargs[:srcfieldfile])-$(active_wavefield)"
            if isfile(filename) == true
                rm(filename)
            end
            iofield[active_wavefield] = open(filename, "w")
            open(kwargs[:compressor][active_wavefield])
        end
    end

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    time1 = time()
    cumtime_io, cumtime_in, cumtime_ex, cumtime_pr = 0.0, 0.0, 0.0, 0.0
    kwargs[:reportinterval] == 0 || @info "nonlinear forward on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"

    dinterp = zeros(Float32, (ntmod_wav - it0 + 1), size(d, 2))

    set_zero_subnormals(true)
    for it = 1:ntmod_wav
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod_wav)
            JopProp2DAcoTTIDenQ_DEO2_FDTD_write_history_nl(kwargs[:ginsu], it, ntmod_wav, time()-time1, cumtime_io, cumtime_in, cumtime_ex, cumtime_pr, wavefields["pcur"], d)
        end

        # propagate and wavefield swap
        cumtime_pr += @elapsed WaveFD.propagateforward!(p)
        wavefields["pcur"],wavefields["pold"] = wavefields["pold"],wavefields["pcur"]
        wavefields["mcur"],wavefields["mold"] = wavefields["mold"],wavefields["mcur"]

        # inject source wavelet
        cumtime_in += @elapsed begin
            WaveFD.injectdata!(wavefields["pcur"], blks_sou_scaled, wavelet_realization, it, kwargs[:nthreads])
            WaveFD.injectdata!(wavefields["pspace"], blks_sou, wavelet_realization, it, kwargs[:nthreads])
        end

        # extract data at receivers
        if length(d) > 0 && it >= it0
            cumtime_ex += @elapsed WaveFD.extractdata!(dinterp, wavefields["pold"], it - it0 + 1, blks_rec, kwargs[:nthreads])
        end

        if it >= it0 && rem(it-it0,itskip) == 0
            # scale spatial derivatives by v^2/b to make them temporal derivatives
            cumtime_pr += @elapsed WaveFD.scale_spatial_derivatives!(p)

            if kwargs[:srcfieldfile] != ""
                cumtime_io += @elapsed for active_wavefield in active_wavefields
                    if kwargs[:isinterior]
                        WaveFD.compressedwrite(iofield[active_wavefield],  kwargs[:compressor][active_wavefield], div(it-it0,itskip)+1, wavefields[active_wavefield], ginsu_interior_range)
                    else
                        WaveFD.compressedwrite(iofield[active_wavefield],  kwargs[:compressor][active_wavefield], div(it-it0,itskip)+1, wavefields[active_wavefield])
                    end
                end
            end
        end
    end
    set_zero_subnormals(false)
    JopProp2DAcoTTIDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod_wav, time()-time1, cumtime_io, cumtime_ex)

    if length(d) > 0
        WaveFD.interpforward!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), d, dinterp)
    end

    if kwargs[:srcfieldfile] != ""
        for active_wavefield in active_wavefields
            close(iofield[active_wavefield])
            close(kwargs[:compressor][active_wavefield])
        end
    end

    free(p)

    nothing
end

function JopProp2DAcoTTIDenQ_DEO2_FDTD_f!(d::AbstractArray, m::AbstractArray{Float32}; kwargs...)
    d .= 0
    isvalid, _chksum = isvalid_srcfieldfile(m, kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-pold", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoTTIDenQ_DEO2_FDTD_nonlinearforward!(d, m; kwargs...)
        kwargs[:chksum][] = _chksum
        kwargs[:srcfieldhost][] = gethostname()
    else
        field = Array{Float32}(undef,size(kwargs[:ginsu], interior=kwargs[:isinterior]))
        local points
        if kwargs[:interpmethod] == :hicks
            points = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], 
                origin(kwargs[:ginsu],interior=kwargs[:isinterior])..., 
                size(kwargs[:ginsu],interior=kwargs[:isinterior])..., kwargs[:rz], kwargs[:rx])
        else
            points = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], 
                origin(kwargs[:ginsu],interior=kwargs[:isinterior])..., 
                size(kwargs[:ginsu],interior=kwargs[:isinterior])..., kwargs[:rz], kwargs[:rx])
        end
        blks = WaveFD.receiver_blocking(points, kwargs[:nthreads])

        iofield = open("$(kwargs[:srcfieldfile])-pold")
        open(kwargs[:compressor]["pold"])
        for it = 1:kwargs[:ntrec]
            WaveFD.compressedread!(iofield, kwargs[:compressor]["pold"], it, field)
            WaveFD.extractdata!(d, field, it, blks, kwargs[:nthreads])
        end

        # interpolate to dtmod and back so amplitudes match data created by nonlinear forward operator
        ntmod = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:ntrec])
        dinterp = zeros(Float32, ntmod, size(d, 2))
        WaveFD.interpadjoint!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 1, WaveFD.LangC(), kwargs[:nthreads]), dinterp, d)
        WaveFD.interpforward!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), d, dinterp)
        close(iofield)
        close(kwargs[:compressor]["pold"])
    end
    d
end

function JopProp2DAcoTTIDenQ_DEO2_FDTD_df!(δd::AbstractArray, δm::AbstractArray; kwargs...)
    nz_ginsu,nx_ginsu=size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu=origin(kwargs[:ginsu])
    p = Prop2DAcoTTIDenQ_DEO2_FDTD(
        nz = nz_ginsu,
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache],
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz],
        dx = kwargs[:dx],
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ],
        qMin = kwargs[:qMin],
        qInterior = kwargs[:qInterior])

    # wave-fields
    pcur,pold = WaveFD.PCur(p),WaveFD.POld(p)

    # earth model
    earth_ginsu = Dict("v"=>WaveFD.V(p), "b"=>WaveFD.B(p), "ϵ"=>WaveFD.Eps(p), "η"=>WaveFD.Eta(p), "f"=>WaveFD.F(p), 
        "cosθ"=>WaveFD.CosTheta(p), "sinθ"=>WaveFD.SinTheta(p))

    # ginsu'd earth model (active-set)
    for prop in keys(kwargs[:active_modelset])
        sub!(earth_ginsu[prop], kwargs[:ginsu], @view(kwargs[:mₒ][:,:,kwargs[:active_modelset][prop]]), extend=true)
    end
    for prop in keys(kwargs[:passive_modelset])
        sub!(earth_ginsu[prop], kwargs[:ginsu], kwargs[:passive_modelset][prop], extend=true)
    end
    earth_ginsu["f"] .= kwargs[:f]

    δm_ginsu = Dict{String,Array{Float32,2}}()
    for prop in keys(kwargs[:active_modelset])
        δm_ginsu[prop] = sub(kwargs[:ginsu], @view(δm[:,:,kwargs[:active_modelset][prop]]), extend=false)
        δm_ginsu[prop] .*= kwargs[:dtrec] / kwargs[:dtmod]
    end

    ginsu_interior_range = interior(kwargs[:ginsu])

    # pre-compute receiver interpolation coefficients
    local points
    if kwargs[:interpmethod] == :hicks
        points = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    else
        points = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    end
    blks = WaveFD.receiver_blocking(points, kwargs[:nthreads])

    # if necessary, re-run the nonlinear forward
    isvalid, _chksum = isvalid_srcfieldfile(kwargs[:mₒ], kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-pold", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoTTIDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), kwargs[:mₒ]; kwargs...)
        kwargs[:srcfieldhost][] = gethostname()
        kwargs[:chksum][] = _chksum
    end

    # local disk file for source wavefield deserialization
    wavefields = Dict{String,Array{Float32,2}}()
    iofields = Dict{String,IOStream}()
    for active_wavefield in kwargs[:active_wavefields]
        iofields[active_wavefield] = open(kwargs[:srcfieldfile]*"-"*active_wavefield)
        open(kwargs[:compressor][active_wavefield])
        wavefields[active_wavefield] = zeros(Float32, nz_ginsu, nx_ginsu)
    end

    ntmod = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:ntrec])

    δdinterp = zeros(Float32, ntmod, size(δd, 2))

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    kwargs[:reportinterval] == 0 || @info "linear forward on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"
    time1 = time()
    cumtime_io, cumtime_ex, cumtime_im, cumtime_pr = 0.0, 0.0, 0.0, 0.0

    set_zero_subnormals(true)
    for it = 1:ntmod
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod)
            JopProp2DAcoTTIDenQ_DEO2_FDTD_write_history_ln(kwargs[:ginsu], it, ntmod, time()-time1, 
                cumtime_io, cumtime_ex, cumtime_im, cumtime_pr, pcur, δdinterp, "forward")
        end

        # propagate and swap wavefields
        cumtime_pr += @elapsed WaveFD.propagateforward!(p)
        pcur,pold = pold,pcur

        if rem(it-1,itskip) == 0
            # read source field from disk
            cumtime_io += @elapsed for active_wavefield in kwargs[:active_wavefields]
                if kwargs[:isinterior]
                    WaveFD.compressedread!(iofields[active_wavefield], kwargs[:compressor][active_wavefield], div(it-1,itskip)+1, wavefields[active_wavefield], ginsu_interior_range)
                else
                    WaveFD.compressedread!(iofields[active_wavefield], kwargs[:compressor][active_wavefield], div(it-1,itskip)+1, wavefields[active_wavefield])
                end
            end

            # born injection
            cumtime_im += @elapsed WaveFD.forwardBornInjection!(p, kwargs[:modeltype], δm_ginsu, wavefields)
        end

        # extract data at receivers
        cumtime_ex += @elapsed WaveFD.extractdata!(δdinterp, pold, it, blks, kwargs[:nthreads])
    end
    set_zero_subnormals(false)
    JopProp2DAcoTTIDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod, time()-time1, 
        cumtime_io, cumtime_ex, cumtime_im)

    WaveFD.interpforward!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), δd, δdinterp)

    for active_wavefield in kwargs[:active_wavefields]
        close(iofields[active_wavefield])
        close(kwargs[:compressor][active_wavefield])
    end

    free(p)

    δd
end

function JopProp2DAcoTTIDenQ_DEO2_FDTD_df′!(δm::AbstractArray, δd::AbstractArray; kwargs...)
    nz_ginsu,nx_ginsu = size(kwargs[:ginsu])
    z0_ginsu,x0_ginsu = origin(kwargs[:ginsu])
    p = Prop2DAcoTTIDenQ_DEO2_FDTD(
        nz = nz_ginsu,
        nx = nx_ginsu,
        nbz = kwargs[:nbz_cache],
        nbx = kwargs[:nbx_cache],
        dz = kwargs[:dz],
        dx = kwargs[:dx],
        dt = kwargs[:dtmod],
        nsponge = kwargs[:nsponge],
        nthreads = kwargs[:nthreads],
        freesurface = kwargs[:freesurface],
        freqQ = kwargs[:freqQ],
        qMin = kwargs[:qMin],
        qInterior = kwargs[:qInterior])

    # wave-fields
    pcur,pold = WaveFD.PCur(p),WaveFD.POld(p)

    # ginsu'd earth model
    earth_ginsu = Dict("v"=>WaveFD.V(p), "b"=>WaveFD.B(p), "ϵ"=>WaveFD.Eps(p), "η"=>WaveFD.Eta(p), "f"=>WaveFD.F(p), 
        "cosθ"=>WaveFD.CosTheta(p), "sinθ"=>WaveFD.SinTheta(p))

    # active model-set
    for prop in keys(kwargs[:active_modelset])
        sub!(earth_ginsu[prop], kwargs[:ginsu], @view(kwargs[:mₒ][:,:,kwargs[:active_modelset][prop]]), extend=true)
    end
    # passive model-set
    for prop in keys(kwargs[:passive_modelset])
        sub!(earth_ginsu[prop], kwargs[:ginsu], kwargs[:passive_modelset][prop], extend=true)
    end
    earth_ginsu["f"] .= kwargs[:f]

    δm_ginsu = Dict{String,Array{Float32,2}}()
    for prop in keys(kwargs[:active_modelset])
        δm_ginsu[prop] = zeros(Float32, nz_ginsu, nx_ginsu)
        if isa(kwargs[:imgcondition], WaveFD.ImagingConditionWaveFieldSeparationMIX)
            δm_ginsu["rtm_$prop"] = zeros(Float32, nz_ginsu, nx_ginsu)
            δm_ginsu["all_$prop"] = zeros(Float32, nz_ginsu, nx_ginsu)
        end
    end

    ginsu_interior_range = interior(kwargs[:ginsu])

    # Get receiver interpolation coefficients
    local points
    if kwargs[:interpmethod] == :hicks
        points = WaveFD.hickscoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    else
        points = WaveFD.linearcoeffs(kwargs[:dz], kwargs[:dx], z0_ginsu, x0_ginsu, nz_ginsu, nx_ginsu, kwargs[:rz], kwargs[:rx])
    end
    for i in eachindex(points)
        ju = points[i].iu
        c = points[i].c * kwargs[:dtmod]^2 * earth_ginsu["v"][ju]^2 / earth_ginsu["b"][ju]
        points[i] = WaveFD.SourcePoint32(ju, points[i].ir, c)
    end
    blks = WaveFD.source_blocking(points, kwargs[:nthreads])

    ntmod = WaveFD.default_ntmod(kwargs[:dtrec], kwargs[:dtmod], kwargs[:ntrec])

    # if necessary, re-run the nonlinear forward
    isvalid, _chksum = isvalid_srcfieldfile(kwargs[:mₒ], kwargs[:srcfieldhost][], kwargs[:srcfieldfile]*"-pold", kwargs[:chksum][])
    if !isvalid
        JopProp2DAcoTTIDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), kwargs[:mₒ]; kwargs...)
        kwargs[:srcfieldhost][] = gethostname()
        kwargs[:chksum][] = _chksum
    end

    # local disk file for source wavefield deserialization
    wavefields = Dict{String,Array{Float32,2}}()
    iofields = Dict{String,IOStream}()
    for active_wavefield in kwargs[:active_wavefields]
        iofields[active_wavefield] = open(kwargs[:srcfieldfile]*"-"*active_wavefield)
        open(kwargs[:compressor][active_wavefield])
        wavefields[active_wavefield] = zeros(Float32, nz_ginsu, nx_ginsu)
    end

    # adjoint interpolation of receiver wavefield
    δdinterp = zeros(Float32, ntmod, size(δd, 2))
    WaveFD.interpadjoint!(WaveFD.interpfilters(kwargs[:dtmod], kwargs[:dtrec], 0, WaveFD.LangC(), kwargs[:nthreads]), δdinterp, δd)

    itskip = round(Int, kwargs[:dtrec]/kwargs[:dtmod])
    kwargs[:reportinterval] == 0 || @info "linear adjoint on $(gethostname()), srcfieldfile=$(kwargs[:srcfieldfile])"
    time1 = time()
    cumtime_io, cumtime_ex, cumtime_im, cumtime_pr = 0.0, 0.0, 0.0, 0.0

    set_zero_subnormals(true)
    for it = ntmod:-1:1
        if kwargs[:reportinterval] != 0 && (it % kwargs[:reportinterval] == 0 || it == ntmod)
            JopProp2DAcoTTIDenQ_DEO2_FDTD_write_history_ln(kwargs[:ginsu], it, ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im, cumtime_pr, pcur, δdinterp, "adjoint")
        end

        # propagate and wavefield swap
        cumtime_pr += @elapsed WaveFD.propagateadjoint!(p)
        pcur,pold = pold,pcur

        # inject receiver data
        cumtime_ex += @elapsed WaveFD.injectdata!(pcur, blks, δdinterp, it, kwargs[:nthreads])

        if rem(it-1,itskip) == 0
            # read source field from disk
            cumtime_io += @elapsed for active_wavefield in kwargs[:active_wavefields]
                if kwargs[:isinterior]
                    WaveFD.compressedread!(iofields[active_wavefield], kwargs[:compressor][active_wavefield], div(it-1,itskip)+1, wavefields[active_wavefield], ginsu_interior_range)
                else
                    WaveFD.compressedread!(iofields[active_wavefield], kwargs[:compressor][active_wavefield], div(it-1,itskip)+1, wavefields[active_wavefield])
                end
            end

            # born accumulation
            cumtime_im += @elapsed WaveFD.adjointBornAccumulation!(p, kwargs[:modeltype], kwargs[:imgcondition], δm_ginsu, wavefields)
        end
    end
    set_zero_subnormals(false)

    for prop in keys(kwargs[:active_modelset])
        if isa(kwargs[:imgcondition], WaveFD.ImagingConditionWaveFieldSeparationMIX)
            weight_all = 1 - kwargs[:RTM_weight]
            weight_short = 2*kwargs[:RTM_weight] - 1
            δm_ginsu[prop] .= (δm_ginsu["all_$prop"]  .* weight_all) .+ (δm_ginsu["rtm_$prop"]  .* weight_short)
        end
        δm_ginsu[prop] .*= kwargs[:dtrec] / kwargs[:dtmod]
    end
    JopProp2DAcoTTIDenQ_DEO2_FDTD_stats(kwargs[:stats], kwargs[:ginsu], ntmod, time()-time1, cumtime_io, cumtime_ex, cumtime_im)

    # undo ginsu
    for prop in keys(kwargs[:active_modelset])
        super!(@view(δm[:,:,kwargs[:active_modelset][prop]]), kwargs[:ginsu], δm_ginsu[prop], accumulate=false)
    end

    for active_wavefield in kwargs[:active_wavefields]
        close(iofields[active_wavefield])
        close(kwargs[:compressor][active_wavefield])
    end

    free(p)

    δm
end

modelindex(F::Jop{T}, key::AbstractString) where {D,R,T<:Jet{D,R,typeof(JopProp2DAcoTTIDenQ_DEO2_FDTD_f!)}} = state(F).active_modelset[key]

function srcillum!(γ, A::T, m::AbstractArray{Float32}; time_mask = IllumOnesVector(Float32)) where {D,R,J<:Jet{D,R,typeof(JopProp2DAcoTTIDenQ_DEO2_FDTD_f!)},T<:Jop{J}}
    s = state(A)
    isvalid, _chksum = isvalid_srcfieldfile(m, s.srcfieldhost[], s.srcfieldfile*"-pold", s.chksum[])
    if !isvalid
        JopProp2DAcoTTIDenQ_DEO2_FDTD_nonlinearforward!(Array{Float32}(undef,0,0), m; s...)
        s.chksum[] = _chksum
        s.srcfieldhost[] = gethostname()
    end
    open(s.compressor["pold"])
    I = srcillum!(γ, s.srcfieldfile*"-pold", s.compressor["pold"], time_mask, s.isinterior, s.ginsu, s.ntrec, s.nthreads)
    close(s.compressor["pold"])
    I
end

function JopProp2DAcoTTIDenQ_DEO2_FDTD_stats(stats, ginsu, it, cumtime_total, cumtime_io, cumtime_ex, cumtime_im=0.0)
    stats["MCells/s"] = megacells_per_second(size(ginsu)..., it-1, cumtime_total)
    stats["%io"] = cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0
    stats["%inject/extract"] = cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0
    stats["%imaging"] = cumtime_total > 0 ? cumtime_im/cumtime_total*100.0 : 0.0
end

Jets.perfstat(J::T) where {D,R,T<:Jet{D,R,typeof(JopProp2DAcoTTIDenQ_DEO2_FDTD_f!)}} = state(J).stats

@inline function JopProp2DAcoTTIDenQ_DEO2_FDTD_write_history_ln(ginsu, it, ntmod, cumtime_total, cumtime_io, cumtime_ex, cumtime_im, cumtime_pr, pcur, d::AbstractArray{T}, mode) where {T}
    jt = occursin("adjoint", mode) ? ntmod-it+1 : it
    kt = cfmt("%5d", jt)
    nt = cfmt("%5d", ntmod)
    mcells = cfmt("%7.2f", megacells_per_second(size(ginsu)..., jt-1, cumtime_total))
    IO = cfmt("%5.2f", cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0)
    EX = cfmt("%5.2f", cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0)
    IM = cfmt("%5.2f", cumtime_total > 0 ? cumtime_im/cumtime_total*100.0 : 0.0)
    PR = cfmt("%5.2f", cumtime_total > 0 ? cumtime_pr/cumtime_total*100.0 : 0.0)
    rmsd = cfmt("%10.4e", length(d) > 0 ? sqrt(norm(d)^2 / length(d)) : zero(T))
    rmsp = cfmt("%10.4e", sqrt(norm(pcur)^2 / length(pcur)))

    @info "Prop2DAcoTTIDenQ_DEO2_FDTD, $mode, time step $kt of $nt $mcells MCells/s (IO=$IO, EX=$EX, IM=$IM, PR=$PR) -- rms d,p; $rmsd $rmsp"
end

@inline function JopProp2DAcoTTIDenQ_DEO2_FDTD_write_history_nl(ginsu, it, ntmod, cumtime_total, cumtime_io, cumtime_in, cumtime_ex, cumtime_pr, pcur, d::AbstractArray{T}) where {T}
    kt = cfmt("%5d", it)
    nt = cfmt("%5d", ntmod)
    mcells = cfmt("%7.2f", megacells_per_second(size(ginsu)..., it-1, cumtime_total))
    IO = cfmt("%5.2f", cumtime_total > 0 ? cumtime_io/cumtime_total*100.0 : 0.0)
    IN = cfmt("%5.2f", cumtime_total > 0 ? cumtime_in/cumtime_total*100.0 : 0.0)
    EX = cfmt("%5.2f", cumtime_total > 0 ? cumtime_ex/cumtime_total*100.0 : 0.0)
    PR = cfmt("%5.2f", cumtime_total > 0 ? cumtime_pr/cumtime_total*100.0 : 0.0)
    rmsd = cfmt("%10.4e", length(d) > 0 ? sqrt(norm(d)^2 / length(d)) : zero(T))
    rmsp = cfmt("%10.4e", sqrt(norm(pcur)^2 / length(pcur)))

    @info "Prop2DAcoTTIDenQ_DEO2_FDTD, nonlinear forward, time step $kt of $nt $mcells MCells/s (IO=$IO, IN=$IN, EX=$EX, PR=$PR) -- rms d,p; $rmsd $rmsp"
end

function Base.close(jet::Jet{D,R,typeof(JopProp2DAcoTTIDenQ_DEO2_FDTD_f!)}) where {D,R}
    rm("$(state(jet).srcfieldfile)-pold", force=true)
    rm("$(state(jet).srcfieldfile)-mold", force=true)
    rm("$(state(jet).srcfieldfile)-pspace", force=true)
    rm("$(state(jet).srcfieldfile)-mspace", force=true)
end
