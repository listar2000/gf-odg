(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k i f c a j e)
(:init 
(handempty)
(ontable k)
(ontable i)
(ontable f)
(ontable c)
(ontable a)
(ontable j)
(ontable e)
(clear k)
(clear i)
(clear f)
(clear c)
(clear a)
(clear j)
(clear e)
)
(:goal
(and
(on k i)
(on i f)
(on f c)
(on c a)
(on a j)
(on j e)
)))