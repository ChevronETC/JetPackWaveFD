using FFTW, Jets, JetPackWaveFD, LinearAlgebra, SpecialFunctions, Statistics, Test, WaveFD

modeltypes = (WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B, WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB)

function make_op(interpmethod, modeltype, fs; comptype = Float32, st = 0.0, wavelet = WaveletCausalRicker(f=5.0), rec2mod=2)
    nsponge = 10
    pad = 20
    nx = 50+2*pad
    ny = 40+2*pad
    nz = 30+2*pad
    nt = 101
    dx,dy,dz,dt = 25.0,25.0,25.0,0.002
    dtrec = dt
    dtmod = dt / rec2mod
    xmin,ymin,zmin = -pad*dx,-pad*dy,-pad*dz
    sx = dx*div(nx,2) + xmin
    sy = dy*div(ny,2) + ymin
    sz = dz
    rz = [dz for iy = 1+pad:ny-pad, ix = 1+pad:nx-pad][:]
    ry = [(iy-1)*dy for iy = 1+pad:ny-pad, ix=1+pad:nx-pad][:] .+ ymin
    rx = [(ix-1)*dx for iy = 1+pad:ny-pad, ix=1+pad:nx-pad][:] .+ xmin

    ρ = 2000.
    b = Float32.(ones(nz, ny, nx) ./ ρ)
    v = 4000 .* ones(Float32,nz,ny,nx)

    local m
    if modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V
        kwargs = (b = b, )
        m = reshape(v, nz, ny, nx, 1)
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB
        kwargs = (nz = nz, ny = ny, nx = nx)
        m = zeros(Float32, nz, ny, nx, 2)
        m[:,:,:,1] .= v
        m[:,:,:,2] .= b
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B
        kwargs = (v = v, )
        m = reshape(b, nz, ny, nx, 1)
    end

    F = JopNlProp3DAcoIsoDenQ_DEO2_FDTD(; kwargs...,
        z0 = zmin, y0 = ymin, x0 = xmin, dz = dz, dy = dy, dx = dx, 
        sz = sz, sy = sy, sx = sx, st = st, rz = rz, ry = ry, rx = rx, 
        dtrec = dtrec, dtmod = dtmod, ntrec = nt, nsponge = nsponge, comptype = comptype, compscale = 1e-4,
        freqQ = 5.0, qMin = 0.1, qInterior = 100.0, wavelet = wavelet, freesurface = fs, 
        reportinterval = 0, interpmethod = interpmethod, isinterior = false)

    m,F
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- close, modeltype=$modeltype" for modeltype in modeltypes
    m₀, F = make_op(:hicks, modeltype, false)
    d = F * m₀
    @test isfile(state(F).srcfieldfile*"-pold")
    @test isfile(state(F).srcfieldfile*"-pspace")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-pold"))
    @test !(isfile(state(F).srcfieldfile*"-pspace"))
    d = F * m₀
    @test isfile(state(F).srcfieldfile*"-pold")
    @test isfile(state(F).srcfieldfile*"-pspace")
    close(F)
    @test !(isfile(state(F).srcfieldfile*"-pold"))
    @test !(isfile(state(F).srcfieldfile*"-p"))

    close(F)
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- perfstat, modeltype=$modeltype" for modeltype in modeltypes
    m₀, F = make_op(:hicks, modeltype, false)

    s = Jets.perfstat(F)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test s["%imaging"] ≈ 0.0

    J = jacobian!(F, m₀)
    d = J * m₀
    s = Jets.perfstat(J)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test isa(s["%imaging"], Float64)

    m = J' * d
    s = Jets.perfstat(J)
    @test isa(s["MCells/s"], Float64)
    @test isa(s["%io"], Float64)
    @test isa(s["%inject/extract"], Float64)
    @test isa(s["%imaging"], Float64)

    close(F)
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- non-causal source injection" begin
    wavelet = WaveletRicker(;f=5.0)
    mₒ, F₁ = make_op(:hicks, WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, false; wavelet, st=-1.0)
    mₒ, F₂ = make_op(:hicks, WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V, false; wavelet, st=-2.0)

    d₁ = F₁ * mₒ
    d₂ = F₂ * mₒ
    @test d₁ ≈ d₂
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- linearization, interpmethod=$interpmethod, fs=$fs, modeltype=$modeltype" for interpmethod in (:linear,:hicks), fs in (true,false), modeltype in modeltypes
    mₒ, F = make_op(interpmethod, modeltype, fs)

    δm = -1 .+ 2 .* rand(domain(F))
    if modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V
        δm[:,:,:,1] .*= 50
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB
        δm[:,:,:,1] .*= 50
        δm[:,:,:,2] .*= 1f-4
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B
        δm[:,:,:,1] .*= 1f-4
    end

    mmask = ones(domain(F))
    isx = (state(F).sx[1] - state(F).x0) / state(F).dx + 1
    isy = (state(F).sy[1] - state(F).y0) / state(F).dy + 1
    isz = (state(F).sz[1] - state(F).z0) / state(F).dz + 1
    srcrad = 1.
    for ix = 1 : size(domain(F), 3)
        for iy = 1 : size(domain(F), 2)
            for iz = 1 : size(domain(F), 1)
                ir = sqrt((ix - isx) ^ 2 + (iy - isy) ^ 2 + (iz - isz) ^ 2)
                if ir <= srcrad
                    mmask[iz, iy, ix, :] .= 0
                end
            end
        end
    end

    μobs, μexp = linearization_test(F, mₒ, δm = δm,
        μ = sqrt.([1.0,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512.0]), mmask=mmask)

    @show μobs, μexp
    @show minimum(abs, μobs - μexp)
    @test minimum(abs, μobs - μexp) < 0.01

    close(F)
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- dot product, interpmethod=$interpmethod, fs=$fs, modeltype=$modeltype" for interpmethod in (:linear,:hicks), fs in (true,false), modeltype in modeltypes
    m₀, F = make_op(interpmethod, modeltype, fs)

    m = -1 .+ 2 .* rand(domain(F))
    if modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_V
        m[:,:,:,1] .*= 50
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_VB
        m[:,:,:,1] .*= 50
        m[:,:,:,2] .*= 0.1
    elseif modeltype == WaveFD.Prop3DAcoIsoDenQ_DEO2_FDTD_Model_B
        m[:,:,:,1] .*= 0.1
    end

    J = jacobian!(F, m₀)
    m = -1 .+ 2*rand(domain(J))
    d = -1 .+ 2*rand(range(J))

    lhs, rhs = dot_product_test(J, m, d)
    err = abs(lhs - rhs) / abs(lhs + rhs)
    @show lhs,rhs,(lhs-rhs)/(lhs+rhs)
    @test lhs ≈ rhs rtol=1e-4
    @test err < 1e-4

    close(F)
end

@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- linearity, interpmethod=$interpmethod, fs=$fs, modeltype=$modeltype" for interpmethod in (:linear,:hicks), fs in (true,false), modeltype in modeltypes
    m₀, F = make_op(interpmethod, modeltype, fs)
    J = jacobian!(F, m₀)
    lhs,rhs = linearity_test(J)
    @test lhs ≈ rhs

    close(F)
end

# note the compression is exercised on the second pass of F * m₀
@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- serialization, modeltype=$modeltype, C=$C" for C in (Float32, UInt32), modeltype in modeltypes
    m₀, F = make_op(:linear, modeltype, false, comptype = C)
    d₁ = F * m₀
    d₂ = F * m₀
    err = sum((d₁-d₂).^2)/sum(d₁.^2 .+ d₂.^2)*2
    @test err < 1e-4

    close(F)
end

@testset "JopProp3DAcoVTIDenQ_DEO2_FDTD -- srcillum, C=$C, modeltype=$modeltype" for C in (Float32, UInt32), modeltype in modeltypes
    m₀, F = make_op(:hicks, modeltype, false, comptype = C)
    d₁ = F * m₀;
    J = jacobian(F, m₀);
    s1 = srcillum(F, m₀);
    s2 = srcillum(J);

    time_mask = ones(Float32, state(F, :ntrec))
    time_mask[1:div(state(F,:ntrec),2)] .= 0
    s3 = srcillum(J; time_mask)
    close(F)
    @test s1 ≈ s2
    @test norm(s3) < norm(s2)
    @test maximum(s1) > 0
end

#=
@testset "JopProp3DAcoIsoDenQ_DEO2_FDTD -- analytic, direct, interpmethod=$interpmethod" for interpmethod in (:hicks,:linear)
    z,y,x,dz,dy,dx,dtrec,dtmod,tmax,sz,sy,sx,c = 2000.0,500.0,2500.0,40.0,40.0,40.0,0.016,0.004,2.0,1000.0,250.0,20.0,1500.0
    nz,ny,nx,nt = round(Int,z/dz)+1,round(Int,y/dy)+1,round(Int,x/dx)+1,round(Int,tmax/dtrec)+1

    fmax = c/6/max(dx,dy,dz)
    fdef = WaveletCausalRicker(f=fmax/2)

    function analytic_result()
        # non-negative frequency axis
        nt_pad = nextprod([2,3,5,7],4*nt)
        nw = div(nt_pad,2)+1
        w = range(0, stop=pi/dtrec, length=nw)

        # wavelet
        f = WaveFD.get(fdef, dtrec*collect(0:nt_pad-1))
        F = rfft(f)

        # Morse and Feshbach, 1953, p. 891
        u_ana = zeros(Float32,nt,nz,ny,nx)
        U = zeros(Complex{Float64},nw)
        for ix = 1:nx, iy = 1:ny, iz = 1:nz
            z = (iz-1)*dz
            y = (iy-1)*dy
            x = (ix-1)*dx
            r = sqrt((z-sz)^2+(y-sy)^2+(x-sx)^2)
            for iw = 2:nw
                if r*w[iw] > eps(Float32)
                    U[iw] = exp(-im*w[iw]/c*r)*F[iw]
                end
            end
            u_ana[:,iz,iy,ix] = irfft(U,nt_pad)[1:nt]
            printfmt("{:.2f}\r", ((ix-1)*ny*nz+(iy-1)*nz+iz-1)/(nz*ny*nx)*100)
        end
        u_ana
    end
    @info "computing analytic result (be patient, can take several minutes)..."
    u_ana = analytic_result()
    @info "...done."

    rx = (dx*[0:nx-1;] * ones(ny)')[:]
    ry = (ones(nx) * (dy*[0:ny-1;])')[:]
    rz = 2*dz*ones(length(rx))

    # Finite difference
    M = JopNlProp3DAcoIsoDenQ_DEO2_FDTD(
        b = (1.0f0/2.0f0)*ones(Float32,nz,ny,nx),
        comptype = UInt32,
        sz = sz,
        sy = sy,
        sx = sx,
        rz = rz,
        ry = ry,
        rx = rx,
        interpmethod = interpmethod,
        dz = dz,
        dy = dy,
        dx = dx,
        dtrec = dtrec,
        dtmod = dtmod,
        ntrec = nt,
        wavelet = fdef,
        nsponge = 200,
        freqQ = 5.0,
        qMin = 0.1,
        qInterior = 100.0,
        freesurface = false,
        reportinterval = 250)

    vp = c*ones(nz,ny,nx)
    @info "computing numerical result (be patient)"
    d_tmp = M*vp
    @info "...done."

    # Read FD field back from disk
    u_fd = zeros(Float32,state(M).ntrec,nz,ny,nx)
    u_snap = zeros(Float32,nz,ny,nx)
    u_snap_ginsu = zeros(Float32,size(state(M).ginsu, interior=false)...)
    io = open("$(state(M).srcfieldfile)-P")

    open(state(M).compressor["P"])
    for i = 1:state(M).ntrec
        WaveFD.compressedread!(io, state(M).compressor["P"], i, u_snap_ginsu)
        super!(u_snap,state(M).ginsu,u_snap_ginsu,interior=false)
        u_fd[i,:,:,:] = u_snap[:,:,:]
    end
    close(state(M).compressor["P"])

    # compute Pearson correlation (https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)
    xx = ones(nz,ny,nx)
    for ix = 1:nx, iy = 1:ny, iz = 1:nz
        xx[iz,iy,ix] = cor(u_fd[:,iz,iy,ix],u_ana[:,iz,iy,ix])
    end
    xx_mean = mean(xx)
    @test xx_mean > .9

    @info "be patient, the garbage collection is probably deleting large scratch-files"

    close(M)
end
=#
