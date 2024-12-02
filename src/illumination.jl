# Jets/WaveFD bridge

struct IllumOnesVector{T} <: AbstractArray{T,1}
    n::Int
    IllumOnesVector(::Type{T}, n) where {T} = new{T}(n)
end
IllumOnesVector(::Type{T}) where {T} = IllumOnesVector(T, typemax(Int))

Base.size(x::IllumOnesVector) = (x.n,)
Base.IndexStyle(::Type{<:IllumOnesVector}) = IndexLinear()
Base.getindex(_::IllumOnesVector{T}, i::Int) where {T} = one(T)
Base.maximum(_::IllumOnesVector{T}) where {T} = one(T)
Base.minimum(_::IllumOnesVector{T}) where {T} = one(T)
Base.extrema(_::IllumOnesVector{T}) where {T} = (one(T),one(T))

# 2D source illumination
function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, nx), filename, compressor, IllumOnesVector(T, ntrec), interior, ginsu, ntrec, nthreads, mask)
end

function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, time_mask::AbstractVector{T}, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, nx), filename, compressor, time_mask, interior, ginsu, ntrec, nthreads, mask)
end

function srcillum!(γ::AbstractArray{T,2}, filename::AbstractString, compressor::WaveFD.Compressor, time_mask::AbstractVector{T}, interior::Bool, ginsu::Ginsu, ntrec::Int, nthreads::Int) where {T}
    io = open(filename)
    field_ginsu = zeros(T, size(ginsu, interior=interior))
    for it = 1:ntrec
        WaveFD.compressedread!(io, compressor, it, field_ginsu)
        srcillum_helper(field_ginsu, time_mask[it])
        super!(γ, ginsu, field_ginsu, accumulate=true, interior=interior)
    end
    close(io)
    γ
end

function srcillum_helper(field_ginsu, time_mask)
    Threads.@threads :static for i in eachindex(field_ginsu)
        field_ginsu[i] *= field_ginsu[i] * time_mask
    end
end

# 3D source illumination
function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, ny::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, ny, nx), filename, compressor, IllumOnesVector(T, ntrec), interior, ginsu, ntrec, nthreads)
end

function srcillum(filename::AbstractString, compressor::WaveFD.Compressor, time_mask::AbstractVector{T}, interior::Bool, ::Type{T}, ginsu::Ginsu, nz::Int, ny::Int, nx::Int, ntrec::Int, nthreads::Int) where {T}
    srcillum!(zeros(T, nz, ny, nx), filename, compressor, time_mask, interior, ginsu, ntrec, nthreads)
end

function srcillum!(γ::AbstractArray{T,3}, filename::AbstractString, compressor::WaveFD.Compressor, time_mask::AbstractVector{T}, interior::Bool, ginsu::Ginsu, ntrec::Int, nthreads::Int) where {T}
    io = open(filename)
    field_ginsu = zeros(T, size(ginsu, interior=interior))
    for it = 1:ntrec
        WaveFD.compressedread!(io, compressor, it, field_ginsu)
        srcillum_helper(field_ginsu, time_mask[it])
        super!(γ, ginsu, field_ginsu, accumulate=true, interior=interior)
    end
    close(io)
    γ
end

@doc """
    srcillum(J[; time_mask])
    srcillum(J, m[; time_mask])
    srcillum!(y, J, m[; time_mask])

Compute and return the source illumination for `Jets` operator `J`. The source illumination 
is defined as the sum of squares of the wavefield amplitudes everywhere in the model over 
the time of the finite difference evolution. 

If `J::Jop` is a `Jets` block operator or distributed block operator, the source 
illuminations from all blocks will be accumulated with a simple sum.

`srcillum(J)` creates and returns an array with the source illumination from `J::JopLn`, 
using the current location defined in the `Jet`.

`srcillum(J, m)` creates and returns an array with the source illumination from `J::Jop` 
for the location `m`.

`srcillum!(y, J, m)` zeros the passed array `y` and then accumulates to the source 
illumination from `J::Jop` at the location `m` into `y`.

A `time_mask` can be used to weight the integration over time.
"""
function srcillum(J::JopLn{<:Jet{<:JetAbstractSpace{T}}}; time_mask=IllumOnesVector(T)) where {T}
    s = zeros(eltype(J), size(domain(J))[1:end-1])
    srcillum!(s, J; time_mask)
end

function srcillum(J::Jop{<:Jet{<:JetAbstractSpace{T}}}, m::AbstractArray{T}; time_mask=IllumOnesVector(T)) where {T}
    s = zeros(eltype(J), size(domain(J))[1:end-1])
    srcillum!(s, J, m; time_mask)
end

@doc (@doc srcillum(J))
srcillum!(γ::AbstractArray, J::Jop{<:Jet{<:JetAbstractSpace{T}}}; time_mask=IllumOnesVector(T)) where {T} = srcillum!(γ, J, jet(J).mₒ; time_mask)

# composite operators, TODO: the try-catch here is a bit of a kludge
function srcillum!(γ::AbstractArray, A::Jet{<:JetAbstractSpace{T},<:JetAbstractSpace,typeof(Jets.JetComposite_f!)}, m::AbstractArray; time_mask=IllumOnesVector(T)) where {T}
    for Aᵢ in state(A).ops
        try
            srcillum!(γ, Aᵢ, m; time_mask)
            break
        catch e
            if length(m) == 0
                rethrow(e)
            end
        end
    end
    γ
end

# block operators
function srcillum!(γ::AbstractArray, A::Jet{<:JetAbstractSpace{T},<:JetAbstractSpace,typeof(Jets.JetBlock_f!)}, m::AbstractArray; time_mask=IllumOnesVector(T)) where {T}
    γ .= 0
    for Aᵢ in state(A).ops
        if length(m) == 0
            srcillum!(γ, Aᵢ, jet(Aᵢ).mₒ; time_mask)
        else
            srcillum!(γ, Aᵢ, m; time_mask)
        end
    end
    γ
end

# distributed block operators
function srcillum!(γ::AbstractArray, A::Jet{<:JetAbstractSpace{T},<:JetAbstractSpace,typeof(DistributedJets.JetDBlock_f!)}, m::AbstractArray; time_mask=IllumOnesVector(T)) where {T}
    γ .= 0
    pids = procs(A)
    γ_partials = ArrayFutures(γ, DistributedJets.addmasterpid(pids))
    time_masks = bcast(time_mask, DistributedJets.addmasterpid(pids))

    _srcillum!(γ_partials, A, m, time_masks) = begin srcillum!(localpart(γ_partials), localpart(A), m; time_mask=localpart(time_masks)); nothing end
    @sync for pid in pids
        @async remotecall_fetch(_srcillum!, pid, γ_partials, A, m, time_tapers)
    end
    reduce!(γ_partials)
end

export srcillum, srcillum!
