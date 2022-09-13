using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Linq;

[RequireComponent(typeof(MeshFilter))]
public class FlowerStem : MonoBehaviour
{
    public int testITERATIOR = 0;
    [Range(3, 32)]
    [SerializeField] int cylinderVertexCount = 8;

    [Range(0.5f,10f)]
    [SerializeField] float stemRadius = 8;

    [Range(0,6)]
    public int maxLeafCount = 3;


    [Range(2, 32)]
    [SerializeField] int edgeRingCount = 8;

    [Range(0, 1)]
    [SerializeField] float tTest = 0;

    [SerializeField] Transform[] controlPoints = new Transform[4];

    Vector3 GetPos(int i) => controlPoints[i].position;

    Mesh mesh;

    private List<Vector3> gizmoVerts = new List<Vector3>();

    [SerializeField] GameObject flowerHead;

    public void loadProperties(StemData data)
    {
        cylinderVertexCount = data._cylinderVertexCount;
        stemRadius = data._stemRadius;
        maxLeafCount = data._leafCount;
        edgeRingCount = data._edgeRingCount;
        controlPoints[0].position = data.startPos;
        controlPoints[1].position = data.tangent1;
        controlPoints[2].position = data.tangent2;
        controlPoints[3].position = data.endPos;
    }

    private void Awake()
    {
        mesh = new Mesh();
        mesh.name = "FlowerStem";

        GetComponent<MeshFilter>().sharedMesh = mesh;


    }

    private void OnDrawGizmos()
    {
        for (int i = 0; i < gizmoVerts.Count; i++)
        {
            //Gizmos.DrawSphere(gizmoVerts[i], 0.02f);
        }

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
    }
    private void Update()
    {
        //GenerateFlowerMesh();
    }

    public void GenerateFlowerMesh()
    {
        mesh.Clear();
        GenerateStemMesh();
        GeneratePistilMesh();
        //GeneratePetalsMeshes();

    }


    private void GeneratePistilMesh()
    {
        OrientedPoint op = GetBezierPoint(1);
        flowerHead.transform.position = op.pos;
        flowerHead.transform.rotation = op.rot;
    }

    private void GenerateStemMesh()
    {
        //VERTICES
        List<Vector3> verts = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();
        gizmoVerts.Clear();

        for (int rings = 0; rings < edgeRingCount; rings++)
        {
            float t = rings / (edgeRingCount - 1f);
            OrientedPoint op = GetBezierPoint(t);

            for (int i = 0; i < cylinderVertexCount; i++)
            {
                float angularStep = i / (float)cylinderVertexCount;
                float angleRad = angularStep * MathfEX.TAU;

                Vector3 randomness = new Vector3(
                    Random.Range(-1, 1),
                    Random.Range(-1, 1),
                    Random.Range(-1, 1)) / 30f;

                verts.Add(op.LocalToWorldPos(MathfEX.GetUnitVectorByAngle(angleRad) * stemRadius + (Vector2)randomness));
                gizmoVerts.Add(op.LocalToWorldPos(MathfEX.GetUnitVectorByAngle(angleRad) * stemRadius));
                normals.Add(op.LocalToWorldVector(Vector3.up));
            }

        }

        //Triangles
        List<int> triIndices = new List<int>();
        for (int ring = 0; ring < edgeRingCount-1; ring++)
        {
            int rootIndex = ring * cylinderVertexCount;
            int rootIndexNext = (ring + 1) * cylinderVertexCount;
            for (int i = 0; i < cylinderVertexCount; i++)
            {
                int currentA = rootIndex + i;
                int currentB = (rootIndex + (rootIndex + i + 1) % cylinderVertexCount);
                int nextA = rootIndexNext + i;
                int nextB = rootIndexNext + ((rootIndexNext + i + 1) % cylinderVertexCount) ;

                triIndices.Add(currentA);
                triIndices.Add(currentB);
                triIndices.Add(nextA);

                triIndices.Add(nextA);
                triIndices.Add(currentB);
                triIndices.Add(nextB);
                testITERATIOR = rootIndex;
            }
        }

        mesh.SetVertices(verts);
        mesh.SetTriangles(triIndices, 0);
        mesh.RecalculateNormals();

    }

    //Returns point on a bezier curve with forward vector
    //pointing towards the end of the curve along it.
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

        return new OrientedPoint(pos, tangent);

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
