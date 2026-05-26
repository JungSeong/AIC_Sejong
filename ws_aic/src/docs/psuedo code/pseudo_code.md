```
// s = {image, force/torque}

Alg distance_delta(s):

  xy_tol = 3 // tolerance, 3mm정도
  
  {delta_x, delta_y, delta_z} = distance_prediction(s)
  a = {delta_x, delta_y, -1}
  
  if abs(delta_x) > xy_tol or abs(delta_y) > xy_tol:
    a = {clip(delta_x), clip(delta_y), 0}
  else:
    a = {clip(delta_x_small), clip(delta_y_small), -insert_step}

  return a

Alg action(s):
  //retry
  fallback_z = 10
  fallback_xy = 50
  {Fx, Fy, Fz} = low_pass_filter(S.F)
  if | Fz | > threshold_z:
    a = {0,0,fallback_z}
  else if | Fx | > threshold_x or | Fy | > threshold_y:
    a = {0,0,fallback_xy}
  else
    a = distance_delta(s)
  return a

Alg policy(s):  
  t = 0
  //aproach and rotation
  a = triangulation_and_rotation(s)
  control(a)

  //align & insert
  while t < time_limit>:
    a = action(s)
    control(a)
    if success(): // detect by CNN 
      return true
    t++;
  return false
```
