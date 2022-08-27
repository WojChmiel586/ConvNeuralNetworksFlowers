using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Timers;
using static UnityEngine.UIElements.UxmlAttributeDescription;

[System.Serializable]
public struct StemData
{
    public int _cylinderVertexCount;
    public float _stemRadius;
    public int _leafCount;
    public int _edgeRingCount;
    public Vector3 startPos, tangent1, tangent2, endPos;
}

[System.Serializable]
public struct HeadData
{
    public int _horizontalLines, _verticalLines;
    public float _radius;
    public Color _petalColour;
}
[System.Serializable]
public struct FlowerData
{
    public StemData stemData;
    public HeadData headData;

    public FlowerData(StemData stemData, HeadData headData)
    {
        this.stemData = stemData;
        this.headData = headData;
    }
}
public class FlowerGenerator : MonoBehaviour
{


    float _roughness = 5f;
    int saveCountIndicator = 0;


    StemData stemData = new StemData();
    HeadData headData = new HeadData();


    public string FlowerPropertiesFileName;
    public FlowerData FlowerData;
    [Header("Stem Properties")]
    public string StemPropertiesFileName;

    private FlowerStem stemScript;

    [Header("Flower Head properties")]
    public string HeadPropertiesFileName;
    private SphereProcGen flowerHeadScript;

    // Start is called before the first frame update
    void Start()
    {
       stemScript = GetComponentInChildren<FlowerStem>();
       flowerHeadScript = GetComponentInChildren<SphereProcGen>();
       //FlowerPropertiesFileName = Application.persistentDataPath + "/flowerdata.json";
       //Debug.Log(FlowerPropertiesFileName = "C:/Users/wojte/OneDrive/Pulpit/FlowerData/" + dateString + " flowerdata.json");
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            GenerateData();
            GenerateFlower();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            SaveFlower();
        }
    }
    public void GenerateData()
    {
        FlowerData.stemData = GenerateStemData();
        FlowerData.headData = GenerateHeadData();
        stemScript.loadProperties(FlowerData.stemData);
        flowerHeadScript.loadProperties(FlowerData.headData);
    }

    StemData GenerateStemData()
    {
        StemData data = new StemData();
        data._cylinderVertexCount = Random.Range(5, 15);
        data._stemRadius = Random.Range(0.03f, 0.1f) + FloatNoise()/10;
        data._leafCount = 8;
        data._edgeRingCount = Random.Range(3, 65);

        data.startPos = Vector3.zero + Vector3Noise();
        data.tangent1 = data.startPos + Vector3Noise() * 2;
        data.tangent2 = data.tangent1 + Vector3Noise() * 2;
        data.endPos = data.tangent2 + Vector3Noise() * 4;

        return data;
    }

    HeadData GenerateHeadData()
    {
        HeadData data = new HeadData();

        data._verticalLines = Random.Range(4, 16);
        data._horizontalLines = Random.Range(4, 16);
        data._radius = FlowerData.stemData._stemRadius + Mathf.Abs(FloatNoise())/10;
        data._petalColour = new Color(Random.Range(0f, 1f), Random.Range(0f,1f), Random.Range(0f, 1f), 1);

        return data;
    }

    float FloatNoise()
    {
        float noise = Random.Range(-_roughness, _roughness)/10f;
        return noise;
    }
    Vector3 Vector3Noise()
    {
        Vector3 noise = new Vector3
               (
               Random.Range(-_roughness, _roughness)/2,
               Mathf.Abs(Random.Range(-_roughness, _roughness))* 2,
               Random.Range(-_roughness, _roughness)/2
               ) / 10f;
        return noise;
    }
    public void GenerateFlower()
    {
        stemScript.GenerateFlowerMesh();
        flowerHeadScript.GenerateHead();
    }

    public void SaveFlower()
    {
        string dateString = System.DateTime.Now.ToString("MM.dd.yyyy");
        FlowerPropertiesFileName = "C:/Users/wojte/OneDrive/Pulpit/FlowerData/" + " ObjectNumber " + saveCountIndicator + " " + dateString + " flowerdata.json";
        FlowerData flower = new FlowerData(stemData,headData);
        string JsonStringFlower = JsonUtility.ToJson(FlowerData,true);

        File.WriteAllText(FlowerPropertiesFileName, JsonStringFlower);
        saveCountIndicator++;

    }

    public void LoadFlower()
    {
        bool missingData = false;
        if (File.Exists(StemPropertiesFileName))
        {
            string fileData = File.ReadAllText(StemPropertiesFileName);
            stemData = JsonUtility.FromJson<StemData>(fileData);
        }
        else
        {
            missingData = true;
        }
        if (File.Exists(HeadPropertiesFileName))
        {
            string fileData = File.ReadAllText(StemPropertiesFileName);
            headData = JsonUtility.FromJson<HeadData>(fileData);
        }
        else
        {
            missingData = true;
        }

        if (File.Exists(FlowerPropertiesFileName))
        {
            FlowerData data = new FlowerData();
            string fileData = File.ReadAllText(FlowerPropertiesFileName);
            data = JsonUtility.FromJson<FlowerData>(fileData);
        }

        if (missingData)
        {
            Debug.LogError("DATA FILE MISSING");
            return;
        }

    }

}
