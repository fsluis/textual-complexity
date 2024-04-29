package ix.common.util;

import com.google.common.collect.HashMultimap;
import ix.common.core.services.ServiceException;
import ix.common.core.xml.Document;
import ix.common.core.xml.XmlNode;
import ix.common.core.xml.XmlParseException;
import ix.common.core.xml.XmlTag;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 18-Dec-2010
 * Time: 13:42:24
 * To change this template use File | Settings | File Templates.
 */
public class ConfigurationNode {
    private String name;
    private HashMultimap<String, ConfigurationNode> children = HashMultimap.create();

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void addChild(ConfigurationNode node) {
        children.put(node.getName(), node);
    }

    public ConfigurationNode getChild(String name) {
        List<ConfigurationNode> nodes = getChildren(name);
        if (nodes.size() > 0)
            return nodes.get(0);
        return null;
    }

    public Collection<ConfigurationNode> getChildren() {
        return children.values();
    }

    public boolean hasChild(String name) {
        return getChild(name) != null;
    }

    public List<ConfigurationNode> getChildren(String name) {
        LinkedList<ConfigurationNode> nodes = new LinkedList<ConfigurationNode>();
        nodes.addAll(children.get(name));
        return nodes;
    }

    public String getValue(String name) {
        if (hasChild(name)) {
            ConfigurationNode node = getChild(name);
            if (node instanceof ConfigurationTextNode)
                return ((ConfigurationTextNode) node).getContent();
        }
        return null;
    }

    public boolean hasValue(String name) {
        return getValue(name) != null;
    }

    /**
     * Checks if a property name has been set, throws an exception otherwise.
     *
     * @param property
     * @throws ServiceException
     */
    public void check(String property) throws ServiceException {
        if(!hasValue(property))
            throw new ServiceException("Please supply a value for the "+property+" property");
    }

    public void check(String[] properties) throws ServiceException {
        for(String property : properties)
            check(property);
    }


    public Map<String,String> getValues(List<ConfigurationNode> nodes) {
        Map<String,String> values = new Hashtable<String,String>();
        for (ConfigurationNode node : nodes)
            if (node instanceof ConfigurationTextNode)
                values.put(node.getName(), ((ConfigurationTextNode) node).getContent());
        return values;
    }

    public List<ConfigurationNode> getGrandChildren(String name) {
        return getAllChildren(name, 0, 2);
    }

    public List<ConfigurationNode> getAllChildren(String name) {
        return getAllChildren(name, 0, Integer.MAX_VALUE);
    }

    private List<ConfigurationNode> getAllChildren(String name, int n, int N) {
        LinkedList<ConfigurationNode> nodes = new LinkedList<ConfigurationNode>();
        nodes.addAll(getChildren(name));
        if (n < N)
            for (ConfigurationNode child : children.values())
                nodes.addAll(child.getAllChildren(name, n + 1, N));
        return nodes;
    }

    /**
     * Flattens the structure to a properties instance (key-value pairs)
     * @return Properties instance
     */
    public Properties toProperties() {
        Map<String, String> values = getValues(getAllChildren());
        Properties properties = new Properties();
        for(Map.Entry<String,String> entry : values.entrySet())
            properties.setProperty(entry.getKey(), entry.getValue());
        return properties;
    }

    public List<ConfigurationNode> getAllChildren() {
        return getAllChildren(0, Integer.MAX_VALUE);
    }

    private List<ConfigurationNode> getAllChildren(int n, int N) {
        LinkedList<ConfigurationNode> nodes = new LinkedList<ConfigurationNode>();
        nodes.addAll(getChildren());
        if (n < N)
            for (ConfigurationNode child : getChildren())
                nodes.addAll(child.getAllChildren(n + 1, N));
        return nodes;
    }

    public static ConfigurationNode parseFromXml(XmlTag configurationTag) throws XmlParseException {
        ConfigurationNode config = new ConfigurationNode();
        config.setName(configurationTag.getName());
        while(configurationTag.hasNextNode()) {
            XmlNode node = configurationTag.getNextNode();
            if (node instanceof Document)
                config.addChild(parseFromXml((Document)node));
            else {
                ConfigurationTextNode child = new ConfigurationTextNode();
                child.setName(node.getName());
                child.setContent(node.getContent());
                config.addChild(child);
            }
        }

        // Hack to merge the always one-child node named "#text" with its parent node.
        if (config.getChildren().size()==1) {
            ConfigurationNode child = config.getChildren().iterator().next();
            if (child instanceof ConfigurationTextNode) {
                ConfigurationTextNode text = ((ConfigurationTextNode)child);
                text.setName(config.getName());
                return text;
            }
        }

        return config;
    }

    public static class ConfigurationTextNode extends ConfigurationNode {
        private String content;

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }
    }

}
