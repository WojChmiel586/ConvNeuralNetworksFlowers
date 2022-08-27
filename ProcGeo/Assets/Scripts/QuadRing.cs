using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class QuadRing : MonoBehaviour
{
    public enum UvProjection
    {
        AngularRadial,
        ProjectZ
    }

    [Range(0.01f, 1f)]
    public float radiusInner;
    [Range(0.01f,2)]
    public float thickness;
    [Range(3, 64)]
    public int angularSegmentCount;
    [SerializeField] UvProjection uvProjection = UvProjection.AngularRadial;

    Mesh mesh;

    float RadiusOuter => radiusInner + thickness;
    int VertexCount => angularSegmentCount * 2;
    //This is a shorcut for this block of code
    //float RadiusOuter
    //{
    //    get
    //    {
    //        return radiusInner + thickness;
    //    }
    //}

    private void OnDrawGizmosSelected()
    {
        GizmosEX.DrawWireCircle(transform.position, transform.rotation, radiusInner,angularSegmentCount);
        GizmosEX.DrawWireCircle(transform.position, transform.rotation, RadiusOuter,angularSegmentCount);
    }


    private void Awake()
    {
        mesh = new Mesh();
        mesh.name = "QuadRing";
        GetComponent<MeshFilter>().sharedMesh = mesh;

    }

    private void Update()
    {
        GenerateMesh();
    }

    void GenerateMesh()
    {
        mesh.Clear();

        int vCount = VertexCount;

        List<Vector3> vertices = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();
        List<Vector2> uvs = new List<Vector2>();

        for (int i = 0; i < angularSegmentCount+ 1; i++)
        {
            float t = i / (float)angularSegmentCount;
            float angRad = t * MathfEX.TAU;

            Vector2 dir = MathfEX.GetUnitVectorByAngle(angRad);

            vertices.Add(dir * RadiusOuter);
            vertices.Add(dir * radiusInner);
            normals.Add(Vector3.forward);
            normals.Add(Vector3.forward);

            switch (uvProjection)
            {
                case UvProjection.AngularRadial:
                    uvs.Add(new Vector2(t, 1));
                    uvs.Add(new Vector2(t, 0));
                    break;
                case UvProjection.ProjectZ:


                    uvs.Add(dir * 0.5f + Vector2.one * 0.5f);
                    uvs.Add(dir * (radiusInner / RadiusOuter) * 0.5f + Vector2.one * 0.5f);
                    break;
                default:
                    break;
            }

        }

        List<int> triangleIndices = new List<int>();

        for (int i = 0; i < angularSegmentCount; i++)
        {
            int rootIndex = i * 2;
            int innerRootIndex = rootIndex + 1;
            int outerNextIndex = rootIndex + 2;
            int innerNextIndex = rootIndex + 3;

            triangleIndices.Add(rootIndex);
            triangleIndices.Add(outerNextIndex);
            triangleIndices.Add(innerNextIndex);

            triangleIndices.Add(rootIndex);
            triangleIndices.Add(innerNextIndex);
            triangleIndices.Add(innerRootIndex);

        }

        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangleIndices, 0);
        mesh.SetNormals(normals);
        mesh.SetUVs(0, uvs);


    }
}
