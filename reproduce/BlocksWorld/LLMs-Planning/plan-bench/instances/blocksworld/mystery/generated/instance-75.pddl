(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e l i a g d)
(:init 
(harmony)
(planet e)
(planet l)
(planet i)
(planet a)
(planet g)
(planet d)
(province e)
(province l)
(province i)
(province a)
(province g)
(province d)
)
(:goal
(and
(craves e l)
(craves l i)
(craves i a)
(craves a g)
(craves g d)
)))