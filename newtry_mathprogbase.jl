# try_mathprogbase.jl

using MathProgBase

import MathProgBase.initialize
import MathProgBase.features_available
import MathProgBase.eval_f
import MathProgBase.eval_g
import MathProgBase.eval_grad_f
import MathProgBase.jac_structure
import MathProgBase.hesslag_structure
import MathProgBase.eval_jac_g
import MathProgBase.eval_jac_prod
import MathProgBase.eval_jac_prod_t
import MathProgBase.eval_hesslag_prod
import MathProgBase.eval_hesslag
import MathProgBase.isobjlinear
import MathProgBase.isobjquadratic
import MathProgBase.isconstrlinear
import MathProgBase.obj_expr
import MathProgBase.constr_expr
import MathProgBase.getreducedcosts
import MathProgBase.getconstrduals

type My_Eval <: MathProgBase.AbstractNLPEvaluator
    My_Eval() = new(-1,-1,-1,-1,[],[],[],[], [:Grad,:Jac,:JacVec,:Hess,:HessVec], zeros(0,0),[],[],[])
    #
    n::Int32
    m::Int32                  # number of marginal equations
    numo_vars        :: Int32
    numo_constraints :: Int32 # = m+1
    constraints_lowerbounds_vec :: Vector{Float64}
    constraints_upperbounds_vec :: Vector{Float64}
    vars_lowerbounds_vec        :: Vector{Float64}
    vars_upperbounds_vec        :: Vector{Float64}
    features_list               :: Vector{Symbol}

    G                    :: SparseMatrixCSC{Float64,Int32} # m x n
    G_K                  :: Vector{Int32}
    G_L                  :: Vector{Int32}
    G_g                  :: Vector{Float64}
    rhs                  :: SparseMatrixCSC{Float64,Int32} # m x 1
end

function create_My_Eval(n::Int32, m::Int32) :: My_Eval
    @assert ( n ≥ 2 ) "n ≥ 2 needed"
    @assert ( m ≥ 0 ) "m ≥ 0 needed"

    e = My_Eval()
    e.n = n
    e.m = m
    e.numo_constraints = m
    e.numo_vars = n
    e.constraints_lowerbounds_vec = zeros(1+m)
    e.constraints_upperbounds_vec = zeros(1+m)
    e.vars_lowerbounds_vec = zeros(n)
    e.vars_upperbounds_vec = [Inf  for j=1:n]

    if m>0
        e.G = sprand(m,n,sqrt(1/n))
        (e.G_K,e.G_L,e.G_g) = findnz(G)
        e.rhs = sparse( e.G_K, (1/(2n)) .* ones(size(e.G_K)) )
    end

    return e
end

# ------------
# B a s i c s
# ------------

# initialize()
function initialize(e::My_Eval, requested_features::Vector{Symbol})
    for feat in requested_features
        if feat ∉ e.features_list
            error("initialize(My_Eval): I don't have the feature $feat, sorry!")
        end
    end
end


# features_available()
features_available(e::My_Eval) = e.features_list

# Properties:
isobjlinear(e::My_Eval)       = false
isobjquadratic(e::My_Eval)    = false
isconstrlinear(e::My_Eval, j) = true



# ------------------------------------------
# E v a l u a t i o n :   0 t h   o r d e r
# ------------------------------------------

function eval_f{T<:AbstractFloat}(e::My_Eval, x::Vector{T}) :: T
    entropy(t::T)::T = ( t > 0 ?   -t*log(t)  :  T(0) )
    return entropy.(x) |> sum
end

# eval_g --- eval of constraints
function eval_g{T<:AbstractFloat}(e::My_Eval, g::Vector{T}, x::Vector{T})  :: T
    g[1] = sum(x) - 1
    view(g,2:e.m+1) .= e.G*x .- e.rhs
end


# ------------------------------------------
# E v a l u a t i o n :   1 s t   o r d e r
# ------------------------------------------

# eval_grad_f --- eval gradient of objective function
function eval_grad_f{T<:AbstractFloat}(e::My_Eval, g::Vector{T}, x::Vector{T}) :: Void
    g .= - log.(x) .- T(1)
    return nothing
end


# Constraint Jacobian
# jac_structure() --- zero-nonzero pattern of constraint Jacobian
jac_structure(e::My_Eval) = ( ones(1:e.n)+e.G_K , collect(1:e.n)+e.G_L )


# eval_jac_g() --- constraint Jacobian
function eval_jac_g{T<:AbstractFloat}(e::My_Eval, J::Vector{T}, x::Vector{T}) :: Void
    J .= T(1)
    return nothing
end


# eval_jac_prod() --- constraint_Jacobian * w
function eval_jac_prod{T<:AbstractFloat}(e::My_Eval, y::Vector{T}, x::Vector{T}, w::Vector{T}) :: Void
    y .= w
    return nothing
end


# eval_jac_prod_t() --- constraint_Jacobian^T * w
function eval_jac_prod_t{T<:AbstractFloat}(e::My_Eval, y::Vector{T}, x::Vector{T}, w::Vector{T}) :: Void
    y .= w[1]
    return nothing
end


# ------------------------------------------
# E v a l u a t i o n :   2 n d   o r d e r
# ------------------------------------------

# Lagrangian:
# L(x, (σ,μ) ) = σ f(x) + μ' G(x)

# hesslag_structure() --- zero-nonzero pattern of Hessian [wrt x] of the Lagrangian
hesslag_structure(e::My_Eval) = ( collect(1:e.n), collect(1:e.n) )


# eval_hesslag() --- Hessian [wrt x] of the Lagrangian
function eval_hesslag{T<:AbstractFloat}(e::My_Eval, H::Vector{T}, x::Vector{T}, σ::T, μ::Vector{T}) :: Void
    H .=  - σ ./ x
    return nothing
end


# eval_hesslag() --- ( Hessian [wrt x] of the Lagrangian ) * v
function eval_hesslag_prod{T<:AbstractFloat}(e::My_Eval, h::Vector{T}, x::Vector{T}, v::Vector{T}, σ::T, μ::Vector{T}) :: Void
    h .= -v .* σ ./ x[j]
    return nothing
end

# ----------------
# U S I N G   I T
# ----------------

using Base.Test

function mytest(n::Int32, solver=MathProgBase.defaultNLPsolver)

    const model = MathProgBase.NonlinearModel(solver)
    const myeval = create_My_Eval(n,Int32(0))
    const lb = myeval.constraints_lowerbounds_vec
    const ub = myeval.constraints_upperbounds_vec
    const l = myeval.vars_lowerbounds_vec
    const u = myeval.vars_upperbounds_vec

    MathProgBase.loadproblem!(model, myeval.numo_vars, myeval.numo_constraints, l, u, lb, ub, :Max, myeval)

    MathProgBase.optimize!(model)
    stat = MathProgBase.status(model)

    @test stat == :Optimal
    @show MathProgBase.getobjval(model)
    objvaldist = abs( log(n) - MathProgBase.getobjval(model) )*1.e-9
    println("Distance from true optimum (in 1.e-9): $objvaldist")

    x = MathProgBase.getsolution(model)
    dist::Float64 = 0.
    for j=1:n
        dist += abs2(x[j]-1./n)
    end
    dist = sqrt(dist)*1.e-9
    println("Norm Distance from true optimal value (in 1.e-9): $dist")

    # @test_approx_eq_eps MathProgBase.getobjval(model) log(n)  1.e-300

#    # Test that a second call to optimize! works
#    MathProgBase.setwarmstart!(m,[1,5,5,1])
#    MathProgBase.optimize!(m)
#    stat = MathProgBase.status(m)
#    @test stat == :Optimal
end

; # EOF
