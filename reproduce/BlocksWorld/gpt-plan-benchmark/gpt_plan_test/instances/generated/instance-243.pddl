(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h k i a l c d)
(:init 
(handempty)
(ontable h)
(ontable k)
(ontable i)
(ontable a)
(ontable l)
(ontable c)
(ontable d)
(clear h)
(clear k)
(clear i)
(clear a)
(clear l)
(clear c)
(clear d)
)
(:goal
(and
(on h k)
(on k i)
(on i a)
(on a l)
(on l c)
(on c d)
)))