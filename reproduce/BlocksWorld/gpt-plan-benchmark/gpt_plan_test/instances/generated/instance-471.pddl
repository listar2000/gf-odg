(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g j c k l e b f d i)
(:init 
(handempty)
(ontable g)
(ontable j)
(ontable c)
(ontable k)
(ontable l)
(ontable e)
(ontable b)
(ontable f)
(ontable d)
(ontable i)
(clear g)
(clear j)
(clear c)
(clear k)
(clear l)
(clear e)
(clear b)
(clear f)
(clear d)
(clear i)
)
(:goal
(and
(on g j)
(on j c)
(on c k)
(on k l)
(on l e)
(on e b)
(on b f)
(on f d)
(on d i)
)))