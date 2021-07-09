package firstpackage

import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.graphx.{Edge, EdgeDirection, EdgeRDD, Graph, TripletFields, VertexId, VertexRDD}
import org.apache.spark.rdd.RDD


object anomalyDetection {
  def main(args: Array[String]): Unit = {
    //creates a SparkSession
    val spark = SparkSession
      .builder
      .appName("first GraphFrame")
      .master("local")
      // .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext


    // load edges from edge file with weights
    val edges: RDD[Edge[Double]] = sc.textFile("src/main/textfiles/directedWeights.txt").map { line =>
      val row = line.split(" ")
      Edge(row(0).toInt, row(1).toInt, row(2).toDouble)
    }

    // load vertices from vertex file with normal/attack types as attributes
    val vertices: RDD[(VertexId,String)] = sc.textFile("src/main/textfiles/vertices.txt").map { line =>
      val row = line.split(", ")
      (row(0).toInt, row(1))
    }

    // val graph2 = Graph.fromEdges(edges, 1)
    val vertcount = vertices.count()
    println(s"number of vertices: $vertcount")


    val graph = Graph(vertices, edges)
    println("graph constructed!")



    // Builds a kNN Graph
    // set k value
    val k = 10

    // aggregate edge weights into source node
    val aggknnEdges: VertexRDD[Array[(VertexId, Double)]] = graph.aggregateMessages[Array[(VertexId, Double)]](
      triplet => { // map
        triplet.sendToSrc(Array((triplet.dstId, triplet.attr)))
      },
      _++_
    )
    // sort edge weights for each node and take k nearest nodes
    val sortedList = aggknnEdges.map { case (v, a) => (v, a.sortWith(_._2 > _._2).take(k))}
    sortedList.foreach { case (v, a) => println(v + ": " + a.mkString)}

    // construct EdgeRDD from sortedList
    val arrknnEdges: RDD[Array[Edge[Double]]] = sortedList.map {
      case (v, a) => val e = a.map {
        case (v1, w) => Edge(v, v1, w)
      } ; e
    }
    val knnEdges: RDD[Edge[Double]] = sc.parallelize(arrknnEdges.reduce(_++_))

    // construct kNN Graph
    val knnGraph = Graph.fromEdges(knnEdges, 1)
    // This moment is dedicated to Willis Duggan



    // get the egonets
    val listOfEgonets: VertexRDD[Array[VertexId]] = knnGraph.aggregateMessages[Array[VertexId]](
      triplet => triplet.sendToSrc(Array(triplet.dstId)), _++_
    )
    listOfEgonets.foreach { case(v, neighbors) => println(v + ": " + neighbors.mkString(" "))}

    // this does the same as above
    // val direction = EdgeDirection.Out
    // val listOfEgonets: VertexRDD[Array[VertexId]] = graph.collectNeighborIds(direction)


    // graph with egonets as vertex property
    val egoGraph = Graph(listOfEgonets, edges)
    println("egonet graph constructed!")


    // get the outdegree of each vertex
    val outDeg: VertexRDD[Int] = egoGraph.outDegrees
    // outDeg.foreach(println)

    /* // does the same thing as above function
    val outDeg: RDD[(VertexId, Int)] = egoGraph.aggregateMessages[Int] (
      triplet => triplet.sendToSrc(1), _+_, TripletFields.EdgeOnly
    )
    */


    // get number of edges in each egonet
    val ec: RDD[(VertexId, Int)] = egoGraph.aggregateMessages[Int] (
      triplet => { // map function
        triplet.dstAttr.foreach(vert =>
          if (triplet.srcAttr contains vert) triplet.sendToSrc(1))
      },
      _+_ // reduce function
    )
    // add outdegree and ec to get total number of edges in each egonet
    val edgeCount: RDD[(VertexId, Int)] = outDeg.join(ec).map {
      case (v, (f, s)) => (v, f + s)
    }



    // Get the total weights of each egonet
    // the weights of outgoing edges of each ego
    val outWeights = egoGraph.aggregateMessages[Double] (
      triplet => triplet.sendToSrc(triplet.attr), _+_
    )

    // store outgoing edge weights in each vertex
    val edgeWeightsInVerts: VertexRDD[Array[(VertexId, Double)]] =
      egoGraph.aggregateMessages[Array[(VertexId, Double)]] (
        triplet => triplet.sendToSrc(Array((triplet.dstId, triplet.attr))), _++_
      )

    // get the weights of edges in egonet not stemming from ego
    val egoPlusWeights = Graph(edgeWeightsInVerts, edges)
    val surroundingWeights = egoPlusWeights.aggregateMessages[Double] (
      triplet => { // map function
        triplet.dstAttr.foreach { case (v1, wgt1) =>
          triplet.srcAttr.foreach { case(v2, _) =>
            if (v1 == v2) triplet.sendToSrc(wgt1)
          }
        }
      },
      _+_ // reduce function
    )

    // combine weights from surrounding and outdegrees for total weight of each egonet
    val totalWeights = outWeights.join(surroundingWeights).map {
      case (v, (w1, w2)) => (v, w1 + w2)
    }


    //pageRank
    val ranks = knnGraph.pageRank(0.001).vertices
    val ranksByType = vertices.join(ranks).map {
      case (id, (user, rank)) => (user, rank)
    }
    //println(ranksByType.collect().mkString("\n"))

    sc.stop()
    spark.stop()
  }
}
