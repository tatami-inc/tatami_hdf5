<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.5">
  <compound kind="file">
    <name>Hdf5CompressedSparseMatrix.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>Hdf5CompressedSparseMatrix_8hpp.html</filename>
    <class kind="class">tatami_hdf5::Hdf5CompressedSparseMatrix</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>Hdf5DenseMatrix.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>Hdf5DenseMatrix_8hpp.html</filename>
    <class kind="class">tatami_hdf5::Hdf5DenseMatrix</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>load_hdf5_matrix.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>load__hdf5__matrix_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
    <member kind="function">
      <type>tatami::CompressedSparseMatrix&lt; row_, Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_ &gt;</type>
      <name>load_hdf5_compressed_sparse_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>acf08ac352e8f0caa63483d0967ea88a3</anchor>
      <arglist>(size_t nr, size_t nc, const std::string &amp;file, const std::string &amp;vals, const std::string &amp;idx, const std::string &amp;ptr)</arglist>
    </member>
    <member kind="function">
      <type>tatami::DenseMatrix&lt;!transpose_, Value_, Index_, ValueStorage_ &gt;</type>
      <name>load_hdf5_dense_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a9c60ce93385d89902f042f3c43fae5bb</anchor>
      <arglist>(const std::string &amp;file, const std::string &amp;name)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>serialize.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>serialize_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
    <member kind="function">
      <type>void</type>
      <name>serialize</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a7e77c93d845d11169165ed14249970fd</anchor>
      <arglist>(Function_ f)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>tatami_hdf5.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>tatami__hdf5_8hpp.html</filename>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>utils.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>utils_8hpp.html</filename>
    <class kind="struct">tatami_hdf5::Hdf5Options</class>
    <namespace>tatami_hdf5</namespace>
  </compound>
  <compound kind="file">
    <name>write_sparse_matrix_to_hdf5.hpp</name>
    <path>/github/workspace/include/tatami_hdf5/</path>
    <filename>write__sparse__matrix__to__hdf5_8hpp.html</filename>
    <class kind="struct">tatami_hdf5::WriteSparseMatrixToHdf5Parameters</class>
    <namespace>tatami_hdf5</namespace>
    <member kind="function">
      <type>void</type>
      <name>write_sparse_matrix_to_hdf5</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a9a0cf7882b3882b88773db1b4a50e4bf</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location, const WriteSparseMatrixToHdf5Parameters &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_sparse_matrix_to_hdf5</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a831a53fbca90181db7cae439bde28b1d</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>tatami_hdf5::Hdf5CompressedSparseMatrix</name>
    <filename>classtatami__hdf5_1_1Hdf5CompressedSparseMatrix.html</filename>
    <templarg>bool row_</templarg>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename CachedValue_</templarg>
    <templarg>typename CachedIndex_</templarg>
    <base>Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>Hdf5CompressedSparseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5CompressedSparseMatrix.html</anchorfile>
      <anchor>af8253455457ec3fdfcfb5be707d92c3e</anchor>
      <arglist>(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr, const Hdf5Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Hdf5CompressedSparseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5CompressedSparseMatrix.html</anchorfile>
      <anchor>a49f8b44c00b690ade286e05407e1e52b</anchor>
      <arglist>(Index_ nr, Index_ nc, std::string file, std::string vals, std::string idx, std::string ptr)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>sparse</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5CompressedSparseMatrix.html</anchorfile>
      <anchor>a7346b17999e5462c5b693367fd534837</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>prefer_rows</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5CompressedSparseMatrix.html</anchorfile>
      <anchor>a388dff91348c274fb58bf09948c886a0</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>tatami_hdf5::Hdf5DenseMatrix</name>
    <filename>classtatami__hdf5_1_1Hdf5DenseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>bool transpose_</templarg>
    <templarg>typename CachedValue_</templarg>
    <base>VirtualDenseMatrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>Hdf5DenseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5DenseMatrix.html</anchorfile>
      <anchor>ad7615b97a6358cba07d061b29a85dfcf</anchor>
      <arglist>(std::string file, std::string name, const Hdf5Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>Hdf5DenseMatrix</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5DenseMatrix.html</anchorfile>
      <anchor>ae57c4a2e0004c2bdb85b543ce06bc15a</anchor>
      <arglist>(std::string file, std::string name)</arglist>
    </member>
    <member kind="function">
      <type>bool</type>
      <name>prefer_rows</name>
      <anchorfile>classtatami__hdf5_1_1Hdf5DenseMatrix.html</anchorfile>
      <anchor>a48c68747e848e3e3625f511d6383664e</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_hdf5::Hdf5Options</name>
    <filename>structtatami__hdf5_1_1Hdf5Options.html</filename>
    <member kind="variable">
      <type>size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__hdf5_1_1Hdf5Options.html</anchorfile>
      <anchor>ad72096e3a430f73a05a6c2e7a6a5d3a3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__hdf5_1_1Hdf5Options.html</anchorfile>
      <anchor>af2a1dfed24de8a01b3c7d9c832e373cd</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_hdf5::WriteSparseMatrixToHdf5Parameters</name>
    <filename>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</filename>
    <member kind="enumeration">
      <type></type>
      <name>StorageLayout</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>ac90ad45811fb4d93ef54a32f023a5f36</anchor>
      <arglist></arglist>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="ac90ad45811fb4d93ef54a32f023a5f36a008f6cdd0c190839e9885cf9f9e2a652">AUTOMATIC</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="ac90ad45811fb4d93ef54a32f023a5f36a829250befeaeea0b203d31fd09a0ced3">COLUMN</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="ac90ad45811fb4d93ef54a32f023a5f36a54c1ed33c810f895d48c008d89f880b7">ROW</enumvalue>
    </member>
    <member kind="enumeration">
      <type></type>
      <name>StorageType</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>aab612e11d67a1447772ea32f42a2291a</anchor>
      <arglist></arglist>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aa008f6cdd0c190839e9885cf9f9e2a652">AUTOMATIC</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aaee9d73311ff0658494edfff14c3ec1e3">INT8</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aaecfc091ed2a607335524c8389cfa41b5">UINT8</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aa5f90af42814c0a419d715d43ae54fd7a">INT16</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aa48d8f1a723d44ff4a87db1bb6c551c62">UINT16</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aa6495adba09844fac8eeb0aba86e6f1bf">INT32</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aa17266551181f69a1b4a3ad5c9e270afc">UINT32</enumvalue>
      <enumvalue file="structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html" anchor="aab612e11d67a1447772ea32f42a2291aafd3e4ece78a7d422280d5ed379482229">DOUBLE</enumvalue>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>data_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>a7c8c64e9fbdb6e24e12727e63cf77682</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>index_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>ade0989560a212da6dedcd56c7608e3bf</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::string</type>
      <name>ptr_name</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>a454ea8bb679e3906a2fb8fde71ce3489</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>StorageLayout</type>
      <name>columnar</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>acfc56828595cd56c598da68c5c2d8541</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>StorageType</type>
      <name>data_type</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>a8799d406493525b368992117dfd86293</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>force_integer</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>a842ef78fe623a54f98c578287975a17b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>StorageType</type>
      <name>index_type</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>a551c36906afbad5495f7fcca8081c44b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>deflate_level</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>acbb4a8f5a508f7aa0372f0c326d410f8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>size_t</type>
      <name>chunk_size</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>ab826ad41c914cc2b92f08dceac019ba3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structtatami__hdf5_1_1WriteSparseMatrixToHdf5Parameters.html</anchorfile>
      <anchor>aa3c2490aaca094b0ff9c6c6d9a10bbe7</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_hdf5</name>
    <filename>namespacetatami__hdf5.html</filename>
    <class kind="class">tatami_hdf5::Hdf5CompressedSparseMatrix</class>
    <class kind="class">tatami_hdf5::Hdf5DenseMatrix</class>
    <class kind="struct">tatami_hdf5::Hdf5Options</class>
    <class kind="struct">tatami_hdf5::WriteSparseMatrixToHdf5Parameters</class>
    <member kind="function">
      <type>tatami::CompressedSparseMatrix&lt; row_, Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_ &gt;</type>
      <name>load_hdf5_compressed_sparse_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>acf08ac352e8f0caa63483d0967ea88a3</anchor>
      <arglist>(size_t nr, size_t nc, const std::string &amp;file, const std::string &amp;vals, const std::string &amp;idx, const std::string &amp;ptr)</arglist>
    </member>
    <member kind="function">
      <type>tatami::DenseMatrix&lt;!transpose_, Value_, Index_, ValueStorage_ &gt;</type>
      <name>load_hdf5_dense_matrix</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a9c60ce93385d89902f042f3c43fae5bb</anchor>
      <arglist>(const std::string &amp;file, const std::string &amp;name)</arglist>
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
      <name>write_sparse_matrix_to_hdf5</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a9a0cf7882b3882b88773db1b4a50e4bf</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location, const WriteSparseMatrixToHdf5Parameters &amp;params)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>write_sparse_matrix_to_hdf5</name>
      <anchorfile>namespacetatami__hdf5.html</anchorfile>
      <anchor>a831a53fbca90181db7cae439bde28b1d</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; *mat, H5::Group &amp;location)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>tatami for HDF5 matrices</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__github_workspace_README</docanchor>
  </compound>
</tagfile>
