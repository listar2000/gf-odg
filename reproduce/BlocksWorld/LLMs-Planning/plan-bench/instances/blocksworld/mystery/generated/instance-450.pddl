(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i a h c l e d)
(:init 
(harmony)
(planet i)
(planet a)
(planet h)
(planet c)
(planet l)
(planet e)
(planet d)
(province i)
(province a)
(province h)
(province c)
(province l)
(province e)
(province d)
)
(:goal
(and
(craves i a)
(craves a h)
(craves h c)
(craves c l)
(craves l e)
(craves e d)
)))