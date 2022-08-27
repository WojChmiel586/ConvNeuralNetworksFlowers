using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class GizmosEX
{
    public static void DrawWireCircle(Vector3 position, Quaternion rotation, float radius, int detail = 32)
    {
        Vector3[] points3D = new Vector3[detail];
        for (int i = 0; i < detail; i++)
        {
            float t = i / (float)detail;
            float angRad = t * MathfEX.TAU;

            Vector2 point2D = MathfEX.GetUnitVectorByAngle(angRad);
            //float angleRad = Mathf.Deg2Rad* (0 + (360/detail)*i);

            point2D *= radius;
            points3D[i] = position + rotation * point2D;
        }

        for (int i = 0; i < detail - 1; i++)
        {
            Gizmos.DrawLine(points3D[i], points3D[i + 1]);
        }
        Gizmos.DrawLine(points3D[detail - 1], points3D[0]);

    }
}
