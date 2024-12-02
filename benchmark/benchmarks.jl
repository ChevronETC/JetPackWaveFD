using BenchmarkTools, JetPackWaveFD, Jets, LinearAlgebra, WaveFD

_nthreads = [8]

const SUITE =  BenchmarkGroup()

n_2D = (z=parse(Int,get(ENV,"2D_NZ","501")), x=parse(Int,get(ENV,"2D_NX","701")))
n_3D = (z=parse(Int,get(ENV,"3D_NZ","101")), y=parse(Int,get(ENV,"3D_NY","102")), x=parse(Int,get(ENV,"3D_NX","103")))

nb_2D = (z=parse(Int,get(ENV,"2D_NBZ","$(n_2D.z)")), x=parse(Int,get(ENV,"2D_NBX","8")))
nb_3D = (z=parse(Int,get(ENV,"3D_NBZ","$(n_3D.z)")), y=parse(Int,get(ENV,"3D_NBY","8")), x=parse(Int,get(ENV,"3D_NBX","8")))

ntmod = WaveFD.default_ntmod(0.004,0.001,10)
SUITE["2DAcoIsoDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoIsoDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function f2diso(nthreads,nz,nx,nbz,nbx,isfieldfile,iscompress)
    JopNlProp2DAcoIsoDenQ_DEO2_FDTD(
        b = ones(Float32,nz,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(nx),
        rx = 10.0*[0:nx-1;],
        dz = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoIsoDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2diso($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["2DAcoVTIDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoVTIDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function f2dvti(nthreads,nz,nx,nbz,nbx,isfieldfile,iscompress)
    JopNlProp2DAcoVTIDenQ_DEO2_FDTD(
        b = ones(Float32,nz,nx),
        ϵ = zeros(Float32,nz,nx),
        η = zeros(Float32,nz,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(nx),
        rx = 10.0*[0:nx-1;],
        dz = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoVTIDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2dvti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["2DAcoTTIDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
SUITE["2DAcoTTIDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_2D.z*n_2D.x,"nbz"=>nb_2D.z,"nbx"=>nb_2D.x,"nthreads"=>_nthreads)])
function f2dtti(nthreads,nz,nx,nbz,nbx,isfieldfile,iscompress)
    JopNlProp2DAcoTTIDenQ_DEO2_FDTD(
        b = ones(Float32,nz,nx),
        ϵ = zeros(Float32,nz,nx),
        η = zeros(Float32,nz,nx),
        θ = zeros(Float32,nz,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(nx),
        rx = 10.0*[0:nx-1;],
        dz = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["2DAcoTTIDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f2dtti($nthreads,$(n_2D.z),$(n_2D.x),$(nb_2D.z),$(nb_2D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_2D.z,2),div(n_2D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["3DAcoIsoDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoIsoDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function f3diso(nthreads,nz,ny,nx,nbz,nby,nbx,isfieldfile,iscompress)
    JopNlProp3DAcoIsoDenQ_DEO2_FDTD(
        b = ones(Float32,nz,ny,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sy = 10.0*div(ny,2),
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(ny*nx),
        ry = [(iy-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        rx = [(ix-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        dz = 10.0,
        dy = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nby_cache = nby,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoIsoDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3diso($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["3DAcoVTIDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoVTIDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function f3dvti(nthreads,nz,ny,nx,nbz,nby,nbx,isfieldfile,iscompress)
    JopNlProp3DAcoVTIDenQ_DEO2_FDTD(
        b = ones(Float32,nz,ny,nx),
        ϵ = zeros(Float32,nz,ny,nx),
        η = zeros(Float32,nz,ny,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sy = 10.0*div(ny,2),
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(ny*nx),
        ry = [(iy-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        rx = [(ix-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        dz = 10.0,
        dy = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nby_cache = nby,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D).y,$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D).y,$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoVTIDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3dvti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["3DAcoTTIDenQ_DEO2_FDTD, no IO, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, forward"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, linear"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, adjoint"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, forward, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, linear, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
SUITE["3DAcoTTIDenQ_DEO2_FDTD, adjoint, compress"] = BenchmarkGroup([Dict("ncells"=>ntmod*n_3D.z*n_3D.y*n_3D.x,"nbz"=>nb_3D.z,"nby"=>nb_3D.y,"nbx"=>nb_3D.x,"nthreads"=>_nthreads)])
function f3dtti(nthreads,nz,ny,nx,nbz,nby,nbx,isfieldfile,iscompress)
    JopNlProp3DAcoTTIDenQ_DEO2_FDTD(
        b = ones(Float32,nz,ny,nx),
        ϵ = zeros(Float32,nz,ny,nx),
        η = zeros(Float32,nz,ny,nx),
        θ = zeros(Float32,nz,ny,nx),
        ϕ = zeros(Float32,nz,ny,nx),
        srcfieldfile = isfieldfile ? tempname() : "",
        comptype = iscompress ? UInt32 : Float32,
        isinterior = true,
        sz = 10.0,
        sy = 10.0*div(ny,2),
        sx = 10.0*div(nx,2),
        rz = 20.0*ones(ny*nx),
        ry = [(iy-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        rx = [(ix-1)*10.0 for iy = 1:ny, ix = 1:nx][:],
        dz = 10.0,
        dy = 10.0,
        dx = 10.0,
        dtmod = 0.001,
        dtrec = 0.004,
        ntrec = 10,
        nbz_cache = nbz,
        nby_cache = nby,
        nbx_cache = nbx,
        freesurface = true,
        nthreads = nthreads,
        reportinterval = 0)
end
for nthreads in _nthreads
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, no IO, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D).y,$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),false,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, forward"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, linear"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, adjoint"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,false); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, forward, compress"]["$nthreads threads"] = @benchmarkable mul!(d,F,m) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, linear, compress"]["$nthreads threads"] = @benchmarkable mul!(δd,J,δm) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=zeros(range(F))) teardown=(close(F))
    SUITE["3DAcoTTIDenQ_DEO2_FDTD, adjoint, compress"]["$nthreads threads"] = @benchmarkable mul!(δm,J',δd) setup=(F=f3dtti($nthreads,$(n_3D.z),$(n_3D.y),$(n_3D.x),$(nb_3D.z),$(nb_3D.y),$(nb_3D.x),true,true); m=1500*ones(domain(F)); d=F*m; J=jacobian!(F,m); δm=zeros(domain(J)); δm[div(n_3D.z,2),div(n_3D.y,2),div(n_3D.x,2)] = 100; δd=J*δm) teardown=(close(F))
end

SUITE["srcillum_helper"] = @benchmarkable JetPackWaveFD.srcillum_helper(x, 1.0f0) setup=(x = rand(Float32,700,600,600))

SUITE