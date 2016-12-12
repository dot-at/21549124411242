
using MathProgBase
;

using Ipopt
mysolver = IpoptSolver()
;

model = MathProgBase.NonlinearModel(mysolver)
;

@show typeof(model)
@show supertype(typeof(model))
;

type My_Eval <: MathProgBase.AbstractNLPEvaluator
    # I'm going to:
    # minimize the natural entropy,
    # subject to being a probability distribution
    # with this range:
    n::Int
    My_Eval(_n) = new(Int(_n))
end
numo_vars(e::My_Eval) :: Int = e.n
numo_constraints(e::My_Eval) :: Int = 1
constraints_lowerbounds_vec(e::My_Eval) :: Vector{Float64} = [1.]
constraints_upperbounds_vec(e::My_Eval) :: Vector{Float64} = [1.]
vars_lowerbounds_vec(e::My_Eval) :: Vector{Float64} = [0.   for j=1:e.n]
vars_upperbounds_vec(e::My_Eval) :: Vector{Float64} = [Inf  for j=1:e.n]
features_list(e::My_Eval) :: Vector{Symbol}  = [:Grad, :Jac, :JacVec, :Hess, :HessVec]
;

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

typealias My_Model typeof(model)
;

function initialize(eval::My_Eval, requested_features::Vector{Symbol})
    for feat in requested_features
        if feat ∉ features_list(eval)
            error("initialize(My_Eval): I don't have the feature $feat, sorry!")
        end
    end
end
;

features_available(eval::My_Eval) = features_list(eval)
;

entropy{T<:AbstractFloat}(t::T)::T = ( t > 0 ?   -t*log(t)  :  T(0) )
eval_f{T<:AbstractFloat}(eval::My_Eval, x::Vector{T}) :: T  = entropy.(x) |> sum
;

function eval_g{T<:AbstractFloat}(eval::My_Eval, g::Vector{T}, x::Vector{T})  :: T
    g[1] = sum(x)
end
;

function eval_grad_f{T<:AbstractFloat}(eval::My_Eval, g::Vector{T}, x::Vector{T}) :: Void
    g .= - log.(x) .- T(1)
    return
end
;

jac_structure(eval::My_Eval) = ( ones(eval.n) , collect(1:eval.n) ) 
;

function eval_jac_g{T<:AbstractFloat}(eval::My_Eval, J::Vector{T}, x::Vector{T}) :: Void
    J .= T(1)
    return
end
;

function eval_jac_prod{T<:AbstractFloat}(eval::My_Eval, y::Vector{T}, x::Vector{T}, w::Vector{T}) :: Void
    y .= w
    return
end
;

function eval_jac_prod_t{T<:AbstractFloat}(eval::My_Eval, y::Vector{T}, x::Vector{T}, w::Vector{T}) :: Void
    y .= w[1]
    return
end
;

hesslag_structure(eval::My_Eval) = ( collect(1:eval.n), collect(1:eval.n) )
;

function eval_hesslag{T<:AbstractFloat}(eval::My_Eval, H::Vector{T}, x::Vector{T}, σ::T, μ::Vector{T}) :: Void
    H .=  - σ ./ x
    return
end
;

function eval_hesslag_prod{T<:AbstractFloat}(eval::My_Eval, h::Vector{T}, x::Vector{T}, v::Vector{T}, σ::T, μ::Vector{T}) :: Void
    h .= -v .* σ ./ x[j]
    return
end
;

isobjlinear(eval::My_Eval) = false
;

isobjquadratic(eval::My_Eval) = false
;

isconstrlinear(eval::My_Eval, i) = true
;

using Base.Test

function mytest(n::Int, solver=MathProgBase.defaultNLPsolver)
    
    const model = MathProgBase.NonlinearModel(solver)
    const myeval = My_Eval(n)
    const lb = constraints_lowerbounds_vec(myeval)
    const ub = constraints_upperbounds_vec(myeval)
    const l = vars_lowerbounds_vec(myeval)
    const u = vars_upperbounds_vec(myeval)

    MathProgBase.loadproblem!(model, numo_vars(myeval), numo_constraints(myeval), l, u, lb, ub, :Max, myeval)

    MathProgBase.optimize!(model)
    stat = MathProgBase.status(model)

    @test stat == :Optimal
    @show MathProgBase.getobjval(model)
    objvaldist = abs( log(n) - MathProgBase.getobjval(model) )*10^9
    println("Distance from true optimum: $objvaldist")
    
    x = MathProgBase.getsolution(model)
    for j=1:n
        @test_approx_eq_eps x[j] 1./n 1.e-10
    end

    @test_approx_eq_eps MathProgBase.getobjval(model) log(n)  1.e-300

#    # Test that a second call to optimize! works
#    MathProgBase.setwarmstart!(m,[1,5,5,1])
#    MathProgBase.optimize!(m)
#    stat = MathProgBase.status(m)
#    @test stat == :Optimal
end
;

mytest(100)


