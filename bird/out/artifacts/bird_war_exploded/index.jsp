<%--
  Created by IntelliJ IDEA.
  User: 何杯水
  Date: 2020/11/23
  Time: 17:18
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
  <head>
    <title>bird</title>
    <form action="${pageContext.request.contextPath }/registerServlet" method="POST" enctype="multipart/form-data">
      姓名: <input type="text" name="name", id="xm"/><br>
      图片: <input type="file" name="photo", id="ph"/><br>
      <input type="submit", value="提交">3
    </form>

  </head>
  <body>

  </body>
</html>
