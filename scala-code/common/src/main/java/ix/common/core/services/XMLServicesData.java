package ix.common.core.services;


import ix.common.util.ConfigurationNode;
import ix.common.core.xml.Document;
import ix.common.core.xml.XmlParseException;
import ix.common.core.xml.XmlTag;

import java.io.InputStream;

/**
 * XML data loader. Please give the xml file name to the constructor, with the xml file having the following structure:<br />
 * &lt;!ELEMENT service-group (service-group*, service*)&gt;<br />
 * &lt;!ATTLIST service-group<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;name ID #IMPLIED<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;file CDATA #IMPLIED<br />
 * &gt;<br />
 *<br />
 * &lt;!ELEMENT service (service-class, init-param*)&gt;<br />
 *<br />
 * &lt;!ELEMENT service-class (#PCDATA)&gt;<br />
 * &lt;!ELEMENT init-param (param-name, param-value)&gt;<br />
 * &lt;!ELEMENT param-name (#PCDATA)&gt;<br />
 * &lt;!ELEMENT param-value (#PCDATA)&gt;<br />
 *<br />
 * The file represents a service-group (servicebranch), which can have a name and has services. Each service consists of
 * a service-class (the classname of which the service is constructed, must implement Service) and init-params
 * (name value pairs). The service-group can contain other service-groups in which case a service-group tag containing
 * the file parameter can be added.<br />
 * <br />
 * Warning: this data handler does not saveData data! It can only load data.
 *
 * @author f
 * @version 1.0
 */
public class XMLServicesData implements ServicesData {
    private InputStream stream;

    public void setStream(InputStream stream) {
        this.stream = stream;
    }

    /**
     * Required param: services-xml, with a full-path filename to the xml file to be read.
     */
    //public void configure(ConfigurationNode config) {
    //    if(config.hasValue("services-xml"))
    //        servicesXML = config.getValue("services-xml");
    //}

    public void load(ServiceBranch branch) throws ServiceException {
        load(branch, stream);
    }

    public boolean save(ServiceLeaf leaf) throws ServiceException {
        throw new ServiceException("Saving is not possible using the XMLServicesData implementation");
    }

    public boolean save(ServiceBranch branch) throws ServiceException {
        throw new ServiceException("Saving is not possible using the XMLServicesData implementation");
    }

    public boolean install() throws ServiceException {
        throw new ServiceException("Installing is not possible using the XMLServicesData implementation");
    }

    /**
     * initialize this ServiceBranch using an xml file. The format for this xml file is specifiek (here)
     *@throws ServiceException an ServiceException is thrown if a parse error occurs in the xml file or the services specified
     *   therein could not be loaded successfully.
     */
    private void load(ServiceBranch branch, InputStream stream) throws ServiceException {
        try {
            Document groupFile = new Document(stream);
            if (!groupFile.getName().equals("service-group")) throw new XmlParseException("service-group tag expected!");
            if(groupFile.hasAttributeValue("name"))
                branch.setName(groupFile.getAttributeValue("name", ""));
            if (branch.getName().equals("")) throw new XmlParseException("service-group/name must be set!");

            /** Read all services */
            while (groupFile.hasNextTag("service")) {
                XmlTag service = groupFile.getNextTag("service");
                if (!service.hasNextTag("class"))
                    throw new XmlParseException("service tag must contain class!");
                String serviceClass = service.getNextTag("class").getValue();
                if (serviceClass == null)
                    throw new XmlParseException("service-class must have a value!");

                // build a Configuration with our read parameters
                ConfigurationNode config = new ConfigurationNode();
                if(service.hasNextTag("configuration"))
                    config = ConfigurationNode.parseFromXml(service.getNextTag("configuration"));

                /** Build a ServiceLeaf */
                ServiceLeaf leaf = new ServiceLeaf(serviceClass, config);
                branch.addLeaf(leaf);
            }

            /** Read all subgroups in the file */
            while (groupFile.hasNextTag("service-group")) {
                XmlTag reference = groupFile.getNextTag("service-group");
                String filename = reference.getAttributeValue("file", "");
                if (filename.equals("")) throw new XmlParseException("service-group/file attribute must have a value!");

                //String directory = xmlFile.getParent();
                // read the subbranch (recursive)
                ServiceBranch child = new ServiceBranch();
                InputStream childStream = Services.findFile(filename);
                load(child, childStream);
                branch.addBranch(child);
            }
        } catch(java.io.IOException ioe) {
            throw new ServiceException(ioe);
        } catch(XmlParseException dpe) {
            throw new ServiceException("File could not be interpreted correctly!", dpe);
        } catch(IllegalArgumentException iae) {
            throw new ServiceException("Filename is illegal!", iae);
        }
    }
}

