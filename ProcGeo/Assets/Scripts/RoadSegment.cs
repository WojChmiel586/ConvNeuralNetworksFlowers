using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Linq;

[RequireComponent(typeof(MeshFilter))]
public class RoadSegment : MonoBehaviour
{

    [SerializeField] Mesh2D shape2D;

    [Range(2,32)]
    [SerializeField] int edgeRingCount = 8;

    [Range(0, 1)]
    [SerializeField] float tTest = 0;

    [SerializeField] Transform[] controlPoints = new Transform[4];

    Vector3 GetPos(int i) => controlPoints[i].position;


    Mesh mesh;

    private void Awake()
    {
        mesh = new Mesh();
        mesh.name = "Segment";



        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    private void Update()
    {
        GenerateMesh();
    }

    void GenerateMesh()
    {
        mesh.Clear();

        //VERTICES
        List<Vector3> verts = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();

        for (int ring = 0; ring < edgeRingCount; ring++)
        {
            float t = ring / (edgeRingCount - 1f);
            OrientedPoint op = GetBezierPoint(t);
            for (int i = 0; i < shape2D.VertexCount; i++)
            {
                verts.Add(op.LocalToWorldPos(shape2D.Vertices[i].point));
                normals.Add(op.LocalToWorldVector(shape2D.Vertices[i].normal));
            }
        }

        //TRIANGLES
        List<int> triIndices = new List<int>();
        for (int ring = 0; ring < edgeRingCount-1; ring++)
        {
            int rootIndex = ring * shape2D.VertexCount;
            int rootIndexNext = (ring + 1) * shape2D.VertexCount;

            for (int line = 0; line < shape2D.LineCount; line+=2)
            {
                int lineIndexA = shape2D.lineIndices[line];
                int lineIndexB = shape2D.lineIndices[line + 1];

                int currentA = rootIndex + lineIndexA;
                int currentB = rootIndex + lineIndexB;
                int nextA = rootIndexNext + lineIndexA;
                int nextB = rootIndexNext + lineIndexB;

                triIndices.Add(currentA);
                triIndices.Add(nextA);
                triIndices.Add(nextB);

                triIndices.Add(currentA);
                triIndices.Add(nextB);
                triIndices.Add(currentB);


            }
        }

        mesh.SetVertices(verts);
        mesh.SetNormals(normals);
        mesh.SetTriangles(triIndices, 0);

    }


/*
    public void OnDrawGizmos()
    {
        for (int i = 0; i < 4; i++)
        {

            Gizmos.DrawSphere(GetPos(i), 0.03f);
        }
        Handles.DrawBezier(
            GetPos(0),
            GetPos(3),
            GetPos(1),
            GetPos(2),
            Color.white,
            EditorGUIUtility.whiteTexture, 1f);




        OrientedPoint testPoint = GetBezierPoint(tTest);
       // Gizmos.DrawSphere(testPoint, 0.03f);
        Handles.PositionHandle(testPoint.pos, testPoint.rot);


        void DrawPoint (Vector2 localPos) 
        {
            Gizmos.DrawSphere(testPoint.LocalToWorldPos(localPos), .15f);
        }

        Vector3[] verts = shape2D.Vertices.Select(v => testPoint.LocalToWorldPos(v.point)).ToArray();

        for (int i = 0; i < shape2D.lineIndices.Length; i+=2)
        {
            Vector3 a = verts[shape2D.lineIndices[i]];
            Vector3 b = verts[shape2D.lineIndices[i+1]];
            Gizmos.DrawLine(a, b);
        }


        //DrawPoint(Vector3.right * 0.2f);
        //DrawPoint(Vector3.right * 0.2f);
        //DrawPoint(Vector3.right * 0.1f);
        //DrawPoint(Vector3.right * -0.2f);
        //DrawPoint(Vector3.right * -0.1f);

        //DrawPoint(Vector3.up * 0.2f);
        //DrawPoint(Vector3.up * 0.1f);
        //DrawPoint(Vector3.up * -0.2f);
        //DrawPoint(Vector3.up * -0.1f);



        Gizmos.color = Color.white;


    }
    */
   OrientedPoint GetBezierPoint(float t)
    {
        Vector3 p0 = GetPos(0);
        Vector3 p1 = GetPos(1);
        Vector3 p2 = GetPos(2);
        Vector3 p3 = GetPos(3);

        Vector3 a = Vector3.Lerp(p0, p1, t);
        Vector3 b = Vector3.Lerp(p1, p2, t);
        Vector3 c = Vector3.Lerp(p2, p3, t);

        Vector3 d = Vector3.Lerp(a, b, t);
        Vector3 e = Vector3.Lerp(b, c, t);

        Vector3 pos = Vector3.Lerp(d, e, t);
        Vector3 tangent = (e - d).normalized;

        return  new OrientedPoint(pos,tangent);


    } 
    Vector3 GetBezierTangent(float t)
    {
        Vector3 p0 = GetPos(0);
        Vector3 p1 = GetPos(1);
        Vector3 p2 = GetPos(2);
        Vector3 p3 = GetPos(3);

        Vector3 a = Vector3.Lerp(p0, p1, t);
        Vector3 b = Vector3.Lerp(p1, p2, t);
        Vector3 c = Vector3.Lerp(p2, p3, t);

        Vector3 d = Vector3.Lerp(a, b, t);
        Vector3 e = Vector3.Lerp(b, c, t);

        return (e - d).normalized;


    }

    float GetApproxLength(int precision = 8)
    {
        Vector3[] points = new Vector3[precision];
        for (int i = 0; i < precision; i++)
        {
            float t = i / (precision - 1);
            points[i] = GetBezierPoint(t).pos;

        }
        float dist = 0;
        for (int i = 0; i < precision - 1; i++)
        {
            Vector3 a = points[i];
            Vector3 b = points[i + 1];
            dist += Vector3.Distance(a, b);
        }

        return dist;
    }

}
