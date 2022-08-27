using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MathfEX
{
    public const float TAU = 6.28318530718f;

    public static Vector2 GetUnitVectorByAngle(float angleRad)
    {
        return new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad));
    }
}
