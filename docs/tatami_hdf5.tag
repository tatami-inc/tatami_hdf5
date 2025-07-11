<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>CompressedSparseMatrix.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>CompressedSparseMatrix_8hpp.html</filename>
    <class kind="struct">tatami_hdf5::CompressedSparseMatrixOptions</class>
    <class kind="class">tatami_hdf5::CompressedSparseMatrix</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>DenseMatrix.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>DenseMatrix_8hpp.html</filename>
    <class kind="struct">tatami_hdf5::DenseMatrixOptions</class>
    <class kind="class">tatami_hdf5::DenseMatrix</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>load_compressed_sparse_matrix.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>load__compressed__sparse__matrix_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>load_dense_matrix.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>load__dense__matrix_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>serialize.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>serialize_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>tatami_hdf5.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>tatami__hdf5_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>utils.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>utils_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>write_compressed_sparse_matrix.hpp</name>
    <path>tatami_hdf5/</path>
    <filename>write__compressed__sparse__matrix_8hpp.html</filename>
    <class kind="struct">tatami_hdf5::WriteCompressedSparseMatrixOptions</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="class">
    <name>tatami_hdf5::CompressedSparseMatrix</name>
    <filename>classtatami__hdf5_1_1CompressedSparseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename CachedValue_</templarg>
    <templarg>typename CachedIndex_</templarg>
    <base>tatami::Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>CompressedSparseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1CompressedSparseMatrix.html</anchorfile>
      <anchor>af59e32e5c129e9666821706195779771</anchor>
      <arglist>(Index_ nrow, Index_ ncol, std::string file_name, std::string value_name, std::string index_name, std::string pointer_name, bool csr, const CompressedSparseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>CompressedSparseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1CompressedSparseMatrix.html</anchorfile>
      <anchor>a1003d0dacaf33924cdb62284f9d8f3f7</anchor>
      <arglist>(Index_ ncsr, Index_ ncol, std::string file_name, std::string value_name, std::string index_name, std::string pointer_name, bool csr)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_hdf5::CompressedSparseMatrixOptions</name>
    <filename>structtatami__hdf5_1_1CompressedSparseMatrixOptions.html</filename>
    <member kind="variable">
      <type>std::size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__hdf5_1_1CompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a961ffcc8dd38d0095219ced5242ac034</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>tatami_hdf5::DenseMatrix</name>
    <filename>classtatami__hdf5_1_1DenseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename CachedValue_</templarg>
    <base>tatami::Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>DenseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1DenseMatrix.html</anchorfile>
      <anchor>ac79ea6b827af72ca406622df388c9327</anchor>
      <arglist>(std::string file, std::string name, bool transpose, const DenseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DenseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1DenseMatrix.html</anchorfile>
      <anchor>a8d5a4a147ad6cd87375f43bc810ce8d0</anchor>
      <arglist>(std::string file, std::string name, bool transpose)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>prefer_rows</name>
      <anchorfile>classtatami__hdf5_1_1DenseMatrix.html</anchorfile>
      <anchor>a992ffa6768fa0d3ebf0eda9fcd9b438d</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_hdf5::DenseMatrixOptions</name>
    <filename>structtatami__hdf5_1_1DenseMatrixOptions.html</filename>
    <member kind="variable">
      <type>std::size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__hdf5_1_1DenseMatrixOptions.html</anchorfile>
      <anchor>a1db8a78128428df5622108188e965cdc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__hdf5_1_1DenseMatrixOptions.html</anchorfile>
      <anchor>a5d24e3dc75b665cb8a54d3d2d1578afb</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_hdf5::WriteCompressedSparseMatrixOptions</name>
    <filename>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</filename>
    <member kind="variable">
      <type>std::string</type>
      <name>data_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a3e84cc9efdefa32f94721d93fd7e4556</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>index_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a451f12a59f08ee3a3b8b493fad71864c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>ptr_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a02bcb589398f16c0b52621bfeae814d3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>WriteStorageLayout</type>
      <name>columnar</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>abfaaaa0eb83834b8e6293e86e7268036</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>WriteStorageType</type>
      <name>data_type</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a54db560e1016dc71f11e5ab2171a54e6</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>force_integer</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a3c2805c165a053161952d62c2088fcc0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>WriteStorageType</type>
      <name>index_type</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a25224e6ff67ab13ccc4ffd24dba7cf54</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>deflate_level</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a3d8c006ce36fbe523e72fb84a1cde8f7</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>hsize_t</type>
      <name>chunk_size</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>a633950d7b6a0d463f32169a9de30db17</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structtatami__hdf5_1_1WriteCompressedSparseMatrixOptions.html</anchorfile>
      <anchor>abf8c63e1057d98410b7cdc741ac4cc08</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_hdf5</name>
    <filename>namespacetatami__hdf5.html</filename>
    <class kind="class">tatami_hdf5::CompressedSparseMatrix</class>
    <class kind="struct">tatami_hdf5::CompressedSparseMatrixOptions</class>
    <class kind="class">tatami_hdf5::DenseMatrix</class>
    <class kind="struct">tatami_hdf5::DenseMatrixOptions</class>
    <class kind="struct">tatami_hdf5::WriteCompressedSparseMatrixOptions</class>
    <member kind="enumeration">
      <type></type>
      <name>WriteStorageLayout</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a6265ac39e6f87118435c505451caa604</anchor>
      <arglist></arglist>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6265ac39e6f87118435c505451caa604a008f6cdd0c190839e9885cf9f9e2a652">AUTOMATIC</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6265ac39e6f87118435c505451caa604a829250befeaeea0b203d31fd09a0ced3">COLUMN</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6265ac39e6f87118435c505451caa604a54c1ed33c810f895d48c008d89f880b7">ROW</enumvalue>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>WriteStorageType</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a6ae3b5a86584169d643af24097ea6cf5</anchor>
      <arglist></arglist>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5a008f6cdd0c190839e9885cf9f9e2a652">AUTOMATIC</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5aee9d73311ff0658494edfff14c3ec1e3">INT8</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5aecfc091ed2a607335524c8389cfa41b5">UINT8</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5a5f90af42814c0a419d715d43ae54fd7a">INT16</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5a48d8f1a723d44ff4a87db1bb6c551c62">UINT16</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5a6495adba09844fac8eeb0aba86e6f1bf">INT32</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5a17266551181f69a1b4a3ad5c9e270afc">UINT32</enumvalue>
      <enumvalue file="namespacetatami__hdf5.html" anchor="a6ae3b5a86584169d643af24097ea6cf5afd3e4ece78a7d422280d5ed379482229">DOUBLE</enumvalue>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>load_compressed_sparse_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a384263136bfe1e251922dadf694903ba</anchor>
      <arglist>(Index_ nr, Index_ nc, const std::string &amp;file, const std::string &amp;vals, const std::string &amp;idx, const std::string &amp;ptr, bool row)</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>load_dense_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>aaaa4d88ae1eafd884f4b225314237ee5</anchor>
      <arglist>(const std::string &amp;file, const std::string &amp;name, bool transpose)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>serialize</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a7e77c93d845d11169165ed14249970fd</anchor>
      <arglist>(Function_ f)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_compressed_sparse_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>abbb2c7427230171df09b7404f3931e02</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location, const WriteCompressedSparseMatrixOptions &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_compressed_sparse_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a8534a867ece6f241802d5cabc9b19d13</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>tatami for HDF5 matrices</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
